"""Analyse one OCR engine's output on the sample; also the Bates-number rules.

Per page: tokens, word-like share, engine confidence and time, Bates-like tokens, whether
the page's own Bates number was read, and footer lines left. Summarised overall, by
scan-quality group and by labeled department.

is_bates_like() decides whether a token looks like a Bates number and matches_bates()
whether it reads as a given page's number, straight, upside down or mirrored; STRAIGHT
and UPSIDE_DOWN map characters OCR confuses with digits. The classifier's text
preparation and the app's pipeline use them.

Writes results/ocr_eda/<engine>/: pages.csv, summary.json (read by the site),
summary.md, worst_pages.csv and charts.

Usage:
    python pipeline/02_ocr/ocr_eda.py --engine A
"""

from __future__ import annotations

import argparse
import difflib
import json
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "pm_sample"
OUT_ROOT = ROOT / "results" / "ocr_eda"

ENGINES = {
    "A": {"name": "Tesseract, 150 dpi (baseline)", "dir": DATA / "ocr",
          "results": "ocr_results.csv", "id_from_path": True},
    "B": {"name": "Tesseract, original resolution", "dir": DATA / "ocr_tess_native",
          "results": "ocr_results.csv", "id_from_path": False},
    "C": {"name": "PaddleOCR, original resolution", "dir": DATA / "ocr_paddle",
          "results": "ocr_results.csv", "id_from_path": False},
    "D": {"name": "Vision-language model", "dir": DATA / "ocr_vlm",
          "results": "ocr_results.csv", "id_from_path": False},
    "E": {"name": "Archive text layer (UCSF)", "dir": DATA / "ocr_ucsf",
          "results": "text_results.csv", "id_from_path": False},
}

NOT_RETRIEVED = {"qnkl0085", "hzgp0030", "xpxk0085"}   # never downloaded
FOOTER_MARKERS = ("industrydocuments", "Source: https")
WORST_N = 10
LOW_CONF = 60

WORD = re.compile(r"^[A-Za-zÄÖÜäöüß]{2,}$")
VOWEL = re.compile(r"[AEIOUYaeiouyÄÖÜäöü]")
PUNCTUATION = ".,;:()'\"!?[]{}"

# Bates heuristic: 9-12 characters, mostly digits or characters OCR confuses with digits
# when a stamp is read straight, upside down or mirrored.
BATES_CHARS = set("0123456789OoDQIl|iZzEASsGbTBgqhLPe")
BATES_MIN_LEN, BATES_MAX_LEN, BATES_MIN_SHARE = 9, 12, 0.8
STRAIGHT = str.maketrans("OoDQIl|iZzASsGbTBgq", "0000111122455667899")
UPSIDE_DOWN = str.maketrans("OoIl|iTZzEhPSsLBGb96", "0111111223445578896" + "6")


def read_text(path: Path) -> str | None:
    """Return a text file's contents, or None if it doesn't exist."""
    return path.read_text(encoding="utf-8", errors="replace") if path.exists() else None


def wordlike_share(tokens: list[str]) -> float:
    """Return the share of tokens that look like words (two or more letters with a vowel)."""
    if not tokens:
        return 0.0
    good = sum(1 for t in tokens if WORD.match(t.strip(PUNCTUATION)) and VOWEL.search(t))
    return round(good / len(tokens), 3)


def is_bates_like(token: str) -> bool:
    """Return whether a token looks like a Bates number (allowing for OCR misreads)."""
    t = token.strip(PUNCTUATION)
    if not BATES_MIN_LEN <= len(t) <= BATES_MAX_LEN:
        return False
    share = sum(c in BATES_CHARS for c in t) / len(t)
    has_digit = any(c.isdigit() for c in t)
    mostly_upper = sum(c.isupper() for c in t) >= len(t) / 2
    return share >= BATES_MIN_SHARE and (has_digit or mostly_upper)


def matches_bates(token: str, bates: str) -> bool:
    """True if the token reads as this page's Bates number, straight or rotated."""
    if not bates:
        return False
    t = token.strip(PUNCTUATION)
    candidates = [t.translate(STRAIGHT), t[::-1].translate(UPSIDE_DOWN), t[::-1].translate(STRAIGHT)]
    return any(difflib.SequenceMatcher(None, c, bates).ratio() >= 0.6 for c in candidates)


def load_results(cfg: dict) -> pd.DataFrame:
    """Read an engine's results table, indexed by ID."""
    path = cfg["dir"] / cfg["results"]
    if not path.exists():
        return pd.DataFrame(columns=["id"]).set_index("id")
    df = pd.read_csv(path, dtype={"id": str})
    if cfg["id_from_path"]:
        df["id"] = df["path"].map(lambda p: Path(str(p).replace("\\", "/")).stem)
    return df.drop_duplicates("id", keep="last").set_index("id")


def build_pages(engine: str) -> pd.DataFrame:
    """Return one row per page with every measure for an engine."""
    cfg = ENGINES[engine]
    manifest = pd.read_csv(DATA / "manifest.csv", dtype=str)
    manifest["position"] = range(1, len(manifest) + 1)
    manifest = manifest[~manifest["id"].isin(NOT_RETRIEVED)]
    results = load_results(cfg)

    rows = []
    for doc_id, position, bates in zip(manifest["id"], manifest["position"], manifest["bates"].fillna("")):
        text = read_text(cfg["dir"] / f"{doc_id}.txt")
        row = {"id": doc_id, "position": position, "read": text is not None}
        if text is not None:
            tokens = text.split()
            bates_tokens = [t for t in tokens if is_bates_like(t)]
            row.update({
                "tokens": len(tokens),
                "wordlike_share": wordlike_share(tokens),
                "bates_like": len(bates_tokens),
                "bates_match": any(matches_bates(t, bates) for t in bates_tokens),
                "footer_lines": sum(any(m in line for m in FOOTER_MARKERS) for line in text.splitlines()),
            })
        if doc_id in results.index:
            for col in ("mean_conf", "seconds", "error"):
                if col in results.columns:
                    row[col] = results.at[doc_id, col]
        rows.append(row)
    pages = pd.DataFrame(rows)

    labels_path = ROOT / "labels" / "label_pm_labels_salo.csv"
    if labels_path.exists():
        labels = pd.read_csv(labels_path, dtype=str, keep_default_na=False).drop_duplicates("path", keep="last")
        labels = labels[labels["outcome"] == "labeled"][["doc_id", "function_id"]]
        pages = pages.merge(labels, left_on="id", right_on="doc_id", how="left").drop(columns="doc_id")
    profile_path = DATA / "page_profile.csv"
    if profile_path.exists():
        pages = pages.merge(pd.read_csv(profile_path, dtype={"id": str, "group": str})[["id", "group"]], on="id", how="left")
    return pages


def summarize(engine: str, pages: pd.DataFrame) -> dict:
    """Return the headline numbers for an engine."""
    read = pages[pages["read"]]
    s = {
        "engine": engine,
        "name": ENGINES[engine]["name"],
        "pages_expected": len(pages),
        "pages_read": int(len(read)),
        "empty_pages": int((read["tokens"] == 0).sum()),
        "tokens_median": float(read["tokens"].median()),
        "tokens_p10": round(float(read["tokens"].quantile(0.10))),
        "tokens_p90": round(float(read["tokens"].quantile(0.90))),
        "wordlike_share_median": float(read["wordlike_share"].median()),
        "pages_mostly_junk": int(((read["wordlike_share"] < 0.5) & (read["tokens"] > 0)).sum()),
        "pages_with_bates_like_token": int((read["bates_like"] > 0).sum()),
        "pages_with_own_bates_found": int(read["bates_match"].sum()),
        "pages_with_footer_left": int((read["footer_lines"] > 0).sum()),
    }
    if "error" in read and read["error"].notna().any():
        s["pages_with_errors"] = int((read["error"].fillna("").astype(str).str.strip() != "").sum())
    if "mean_conf" in read and read["mean_conf"].notna().any():
        conf = pd.to_numeric(read["mean_conf"], errors="coerce")
        s["conf_median"] = float(conf.median())
        s[f"pages_under_{LOW_CONF}_conf"] = int((conf < LOW_CONF).sum())
    if "seconds" in read and read["seconds"].notna().any():
        sec = pd.to_numeric(read["seconds"], errors="coerce")
        s["seconds_mean"] = round(float(sec.mean()), 2)
        s["seconds_total_min"] = round(float(sec.sum()) / 60, 1)
    for col, key in (("group", "by_group"), ("function_id", "by_department")):
        if col in read:
            g = read.dropna(subset=[col]).groupby(col)
            s[key] = {str(k): {"pages": int(len(v)),
                               "tokens_median": float(v["tokens"].median()),
                               "wordlike_share_median": float(v["wordlike_share"].median()),
                               "empty_pages": int((v["tokens"] == 0).sum())}
                      for k, v in g}
    return s


def worst_pages(pages: pd.DataFrame) -> pd.DataFrame:
    """Return the weakest pages by readability."""
    read = pages[pages["read"]]
    cols = [c for c in ("position", "id", "tokens", "wordlike_share", "mean_conf", "group", "function_id") if c in read]
    return read.sort_values(["wordlike_share", "tokens"]).head(WORST_N)[cols]


def write_markdown(s: dict, worst: pd.DataFrame, path: Path) -> None:
    """Write the readable summary."""
    lines = [f"# OCR engine {s['engine']}: {s['name']}", "",
             "| Measure | Value |", "|---|---|"]
    for k, v in s.items():
        if not isinstance(v, dict) and k not in ("engine", "name"):
            lines.append(f"| {k.replace('_', ' ')} | {v} |")
    for key, title in (("by_group", "By scan-quality group"), ("by_department", "By labeled department")):
        if key in s:
            lines += ["", f"## {title}", "", "| | pages | median tokens | median word-like share | empty |",
                      "|---|---|---|---|---|"]
            lines += [f"| {k} | {v['pages']} | {v['tokens_median']:.0f} | {v['wordlike_share_median']:.3f} | "
                      f"{v['empty_pages']} |" for k, v in s[key].items()]
    lines += ["", f"## Weakest {WORST_N} pages", "", worst.to_markdown(index=False)]
    path.write_text("\n".join(lines), encoding="utf-8")


def write_charts(engine: str, pages: pd.DataFrame, out: Path) -> None:
    """Write the charts (needs matplotlib)."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; charts skipped (pip install matplotlib)")
        return
    read = pages[pages["read"]]
    name = ENGINES[engine]["name"]
    charts = [("tokens", "Tokens per page", 40), ("wordlike_share", "Word-like share per page", 20)]
    if "mean_conf" in read and read["mean_conf"].notna().any():
        charts.append(("mean_conf", "Mean confidence per page", 20))
    for col, title, bins in charts:
        fig, ax = plt.subplots(figsize=(6, 3.5))
        ax.hist(pd.to_numeric(read[col], errors="coerce").dropna(), bins=bins, color="#5b59a6")
        ax.set_title(f"{engine}: {title}")
        ax.set_xlabel(title.split(" per")[0])
        ax.set_ylabel("Pages")
        fig.suptitle(name, fontsize=8, y=0.98)
        fig.tight_layout()
        fig.savefig(out / f"{col}_hist.png", dpi=120)
        plt.close(fig)


def main() -> None:
    """Analyse one engine and save the results."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--engine", required=True, choices=sorted(ENGINES))
    args = parser.parse_args()

    if not ENGINES[args.engine]["dir"].exists():
        raise SystemExit(f"No output found for engine {args.engine} in {ENGINES[args.engine]['dir']}")

    out = OUT_ROOT / args.engine
    out.mkdir(parents=True, exist_ok=True)
    pages = build_pages(args.engine)
    pages.to_csv(out / "pages.csv", index=False)
    s = summarize(args.engine, pages)
    (out / "summary.json").write_text(json.dumps(s, indent=1), encoding="utf-8")
    worst = worst_pages(pages)
    worst.to_csv(out / "worst_pages.csv", index=False)
    try:
        write_markdown(s, worst, out / "summary.md")
    except ImportError:   # DataFrame.to_markdown needs the 'tabulate' package
        print("tabulate not installed; summary.md skipped (pip install tabulate)")
    write_charts(args.engine, pages, out)

    print(f"\nEngine {args.engine}: {s['name']}")
    for k, v in s.items():
        if not isinstance(v, dict) and k not in ("engine", "name"):
            print(f"  {k.replace('_', ' '):32} {v}")
    for key, title in (("by_group", "By scan-quality group"), ("by_department", "By labeled department")):
        if key in s:
            print(f"\n  {title}")
            print(pd.DataFrame(s[key]).T.to_string())
    print(f"\n  Weakest {WORST_N} pages")
    print(worst.to_string(index=False))
    print(f"\nWritten to {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
