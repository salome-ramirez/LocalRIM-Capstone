"""Compare the five OCR engines on the sample: the tables the engine choice is made from.

Combines, for every engine, the measures on all 1,400 pages and the blind review of the
22 test pages, broken down by scan-quality group.

Engines:
    A  Tesseract, 150-dpi pages                 data/pm_sample/ocr/
    B  Tesseract, scan resolution               data/pm_sample/ocr_tess_native/
    C  PaddleOCR                                data/pm_sample/ocr_paddle/
    D  GLM-OCR through Ollama                   data/pm_sample/ocr_vlm/
    E  the archive's own text layer             data/pm_sample/ocr_ucsf/  (reference only)

Measures on all pages: tokens, empty pages, word-like share (tokens of two or more
letters with a vowel; a readability proxy) and pages where a piece of the archive's
source footer survived. Blind review (results/ocr_review_ratings.csv): points from 4 (all
or nearly all) to 0 (none), and each page's rank as a 0-1 score; the candidates-only
score drops E and re-ranks the rest. Results are given per assigned scan group, because
the test pages over-weight the small, hard groups.

Writes results/ocr_compare.csv (per page) and, in results/ocr_compare/: pages_by_group,
footer_fragments, review_overall, review_by_group, group_check and funsd (.csv).

Usage:
    python -m pipeline.02_ocr.ocr_compare
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import pandas as pd

from pipeline.common import ROOT, TRAIN_LABELS

DATA = ROOT / "data" / "pm_sample"
OUTPUTS = ROOT / "results"
DEFAULT_OUT = OUTPUTS / "ocr_compare.csv"
TABLES = OUTPUTS / "ocr_compare"
REVIEW_FILE = OUTPUTS / "ocr_review_ratings.csv"
GROUP_CHECK_FILE = OUTPUTS / "scan_group_check.csv"

ENGINES = {
    "A": DATA / "ocr",
    "B": DATA / "ocr_tess_native",
    "C": DATA / "ocr_paddle",
    "D": DATA / "ocr_vlm",
    "E": DATA / "ocr_ucsf",
}
REFERENCE_ENGINES = {"E"}  # reference, not a candidate

# Documents whose download returned a web page instead of the scan.
NOT_RETRIEVED = {"qnkl0085", "hzgp0030", "xpxk0085"}

WORD = re.compile(r"^[A-Za-zÄÖÜäöüß]{2,}$")
VOWEL = re.compile(r"[AEIOUYaeiouyÄÖÜäöü]")
PUNCTUATION = ".,;:()'\"!?"
# Pieces of "Source: https://www.industrydocuments.ucsf.edu/docs/<id>" that survive filtering
FOOTER_FRAGMENT = re.compile(r"(?i)(industry\s*doc|ucsf\.edu|ucsf|/docs/|\bsource\s*:|\bsou\.\.)")

POINTS = {"All or nearly all": 4, "Most": 3, "About half": 2, "Few": 1, "None": 0}
RATINGS = ["Good", "Usable", "Poor"]
REVIEW_COLUMNS = ["id", "position", "engine", "reading_slot", "rating", "words_found", "spelling",
                  "issues", "note", "rank", "n_readings", "saved_at"]


# 1,400-page measures

def text_stats(path: Path) -> tuple[int | None, float | None, bool | None]:
    """Return (token count, word-like share, footer fragment present) for one text file."""
    if not path.exists():
        return None, None, None
    text = path.read_text(encoding="utf-8", errors="replace")
    tokens = text.split()
    if not tokens:
        return 0, 0.0, False
    wordlike = sum(1 for t in tokens if WORD.match(t.strip(PUNCTUATION)) and VOWEL.search(t))
    return len(tokens), round(wordlike / len(tokens), 3), bool(FOOTER_FRAGMENT.search(text))


def load_tesseract_results(path: Path, id_from_path: bool = False) -> pd.DataFrame:
    """Load a Tesseract results table indexed by document ID."""
    df = pd.read_csv(path, dtype={"id": str})
    if id_from_path:
        df["id"] = df["path"].map(lambda p: Path(str(p).replace("\\", "/")).stem)
    return df.drop_duplicates("id", keep="last").set_index("id")


def load_groups() -> pd.DataFrame | None:
    """Return the scan group of every page."""
    path = DATA / "page_profile.csv"
    if not path.exists():
        return None
    return pd.read_csv(path, dtype={"id": str, "group": str})[["id", "group"]]


def build_table() -> pd.DataFrame:
    """Return one row per page with the measures for every engine."""
    manifest = pd.read_csv(DATA / "manifest.csv", dtype=str)
    manifest["position"] = range(1, len(manifest) + 1)
    manifest = manifest[~manifest["id"].isin(NOT_RETRIEVED)]

    a = load_tesseract_results(ENGINES["A"] / "ocr_results.csv", id_from_path=True)
    b = load_tesseract_results(ENGINES["B"] / "ocr_results.csv")

    rows = []
    for doc_id, position in zip(manifest["id"], manifest["position"]):
        row = {"id": doc_id, "position": position}
        for tag, folder in ENGINES.items():
            (row[f"{tag}_tokens"], row[f"{tag}_wordlike"],
             row[f"{tag}_footer"]) = text_stats(folder / f"{doc_id}.txt")
        row["A_conf"] = a["mean_conf"].get(doc_id)
        row["A_seconds"] = a["seconds"].get(doc_id)
        row["B_conf"] = b["mean_conf"].get(doc_id)
        row["B_seconds"] = b["seconds"].get(doc_id)
        rows.append(row)
    table = pd.DataFrame(rows)

    labels_path = TRAIN_LABELS
    if labels_path.exists():
        labels = pd.read_csv(labels_path, dtype=str, keep_default_na=False)
        labels = labels.drop_duplicates("path", keep="last")
        labels = labels[labels["outcome"] == "labeled"][["doc_id", "function_id"]]
        table = table.merge(labels, left_on="id", right_on="doc_id", how="left").drop(columns="doc_id")

    groups = load_groups()
    if groups is not None:
        table = table.merge(groups, on="id", how="left")
    return table


def summarize(table: pd.DataFrame) -> pd.DataFrame:
    """Return the headline measures per engine."""
    summary = {}
    for tag in ENGINES:
        tokens, wordlike = table[f"{tag}_tokens"], table[f"{tag}_wordlike"]
        summary[tag] = {
            "pages": int(tokens.notna().sum()),
            "median tokens": int(tokens.median()) if tokens.notna().any() else None,
            "empty pages": int((tokens == 0).sum()),
            "word-like share (median)": round(float(wordlike.median()), 3) if wordlike.notna().any() else None,
            "pages with footer fragments": int(table[f"{tag}_footer"].fillna(False).astype(bool).sum()),
        }
    for tag in ("A", "B"):
        summary[tag]["median confidence"] = round(float(table[f"{tag}_conf"].median()), 1)
        summary[tag]["seconds/page (mean)"] = round(float(table[f"{tag}_seconds"].mean()), 2)
    return pd.DataFrame(summary)


def pages_by_group(table: pd.DataFrame) -> pd.DataFrame:
    """Return median tokens and word-like share per scan group and engine."""
    rows = []
    for group, part in table.groupby("group"):
        for tag in ENGINES:
            rows.append({
                "group": group, "engine": tag, "pages": len(part),
                "median_tokens": part[f"{tag}_tokens"].median(),
                "median_wordlike": round(float(part[f"{tag}_wordlike"].median()), 3),
                "empty_pages": int((part[f"{tag}_tokens"] == 0).sum()),
            })
    return pd.DataFrame(rows)


def footer_fragments(table: pd.DataFrame) -> pd.DataFrame:
    """Return every page and engine where a footer piece survived."""
    rows = [{"id": r["id"], "engine": tag}
            for _, r in table.iterrows() for tag in ENGINES
            if pd.notna(r[f"{tag}_footer"]) and bool(r[f"{tag}_footer"])]
    return pd.DataFrame(rows, columns=["id", "engine"])


# Blind review

def read_review() -> pd.DataFrame | None:
    """Return the ratings, newest save per page and engine.

    Rows written before the rank columns existed (10 fields) and after (12) are read by
    their own field count.
    """
    if not REVIEW_FILE.exists():
        return None
    with REVIEW_FILE.open(newline="", encoding="utf-8") as f:
        rows = list(csv.reader(f))
    header, records = rows[0], []
    for row in rows[1:]:
        if not row:
            continue
        cols = header if len(row) == len(header) else REVIEW_COLUMNS if len(row) == len(REVIEW_COLUMNS) else None
        if cols is None:
            raise ValueError(f"Unexpected row length {len(row)} in {REVIEW_FILE.name}")
        records.append(dict(zip(cols, row)))
    review = pd.DataFrame(records).reindex(columns=REVIEW_COLUMNS, fill_value="").fillna("")
    review = review.drop_duplicates(subset=["id", "engine"], keep="last")

    for col in ("words_found", "spelling"):
        review[f"{col}_points"] = review[col].map(POINTS)
    review["rank"] = pd.to_numeric(review["rank"], errors="coerce")
    review["n"] = review.groupby("id")["rank"].transform("count")
    review["rank_score"] = (review["rank"] - 1) / (review["n"] - 1)

    cand = review[~review["engine"].isin(REFERENCE_ENGINES) & review["rank"].notna()].copy()
    cand["crank"] = cand.groupby("id")["rank"].rank(method="first")
    cand["cn"] = cand.groupby("id")["crank"].transform("count")
    review["rank_score_candidates"] = ((cand["crank"] - 1) / (cand["cn"] - 1)).reindex(review.index)

    groups = load_groups()
    if groups is not None:
        review = review.merge(groups, on="id", how="left")
    return review


def review_summary(review: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    """Return rating counts, mean points and rank scores per engine (and group)."""
    def one(part: pd.DataFrame) -> pd.Series:
        counts = part["rating"].value_counts()
        return pd.Series({
            "pages": part["id"].nunique(),
            **{r.lower(): int(counts.get(r, 0)) for r in RATINGS},
            "words_found": round(part["words_found_points"].mean(), 2),
            "spelling": round(part["spelling_points"].mean(), 2),
            "rank_score": round(part["rank_score"].mean(), 2),
            "rank_score_candidates": round(part["rank_score_candidates"].mean(), 2),
        })
    return review.groupby(by).apply(one).reset_index()


FUNSD_FILES = {
    "A": OUTPUTS / "funsd_ocr_eval.csv",
    "C": OUTPUTS / "funsd_ocr_eval_C.csv",
    "D": OUTPUTS / "funsd_ocr_eval_D.csv",
}


def funsd_summary() -> pd.DataFrame | None:
    """Return FUNSD accuracy per engine from the per-form files.

    Pooled word recall and WER weight each form by its number of true words.
    """
    rows = []
    for engine, path in FUNSD_FILES.items():
        if not path.exists():
            continue
        df = pd.read_csv(path)
        words = df["true_words"]
        conf = pd.to_numeric(df.get("mean_conf"), errors="coerce") if "mean_conf" in df else None
        rows.append({
            "engine": engine,
            "forms": len(df),
            "true_words": int(words.sum()),
            "word_recall": round(float((df["word_recall"] * words).sum() / words.sum()), 3),
            "wer": round(float((df["wer"] * words).sum() / words.sum()), 3),
            "wer_median_form": round(float(df["wer"].median()), 3),
            "seconds_per_form": round(float(pd.to_numeric(df["seconds"], errors="coerce").mean()), 2),
            "conf_vs_recall_spearman": (round(float(conf.rank().corr(df["word_recall"].rank())), 2)
                                        if conf is not None and conf.notna().any() else None),
        })
    return pd.DataFrame(rows) if rows else None


def group_check() -> pd.DataFrame | None:
    """Return the scan group check per assigned group."""
    if not GROUP_CHECK_FILE.exists():
        return None
    checks = pd.read_csv(GROUP_CHECK_FILE, dtype=str, keep_default_na=False)
    checks = checks.drop_duplicates(subset=["id"], keep="last")
    table = pd.crosstab(checks["assigned_group"], checks["answer"])
    table = table.reindex(columns=["Correct", "Wrong", "Unsure"], fill_value=0)
    wrong = checks[checks["answer"] == "Wrong"]
    table["moved_to"] = wrong.groupby("assigned_group")["reviewer_group"].agg(
        lambda s: ", ".join(sorted(s))).reindex(table.index).fillna("")
    return table.reset_index()



def main() -> None:
    """Build and save every comparison table."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="per-page CSV output path")
    args = parser.parse_args()

    pd.set_option("display.width", 160)
    pd.set_option("display.max_columns", 20)
    TABLES.mkdir(parents=True, exist_ok=True)

    table = build_table()
    table.to_csv(args.out, index=False)
    print("\nAll 1,400 pages")
    print(summarize(table).to_string())

    if "group" in table:
        by_group = pages_by_group(table)
        by_group.to_csv(TABLES / "pages_by_group.csv", index=False)
        print("\nMedian word-like share by scan group")
        print(by_group.pivot(index="group", columns="engine", values="median_wordlike").to_string())
    footer_fragments(table).to_csv(TABLES / "footer_fragments.csv", index=False)

    review = read_review()
    if review is not None:
        overall = review_summary(review, ["engine"])
        overall.to_csv(TABLES / "review_overall.csv", index=False)
        print("\nBlind review, all 22 test pages (E is a reference, D-51)")
        print(overall.to_string(index=False))
        if "group" in review:
            by_group = review_summary(review, ["group", "engine"])
            by_group.to_csv(TABLES / "review_by_group.csv", index=False)
            print("\nBlind review by scan group")
            print(by_group.to_string(index=False))

    funsd = funsd_summary()
    if funsd is not None:
        funsd.to_csv(TABLES / "funsd.csv", index=False)
        print("\nFUNSD accuracy (50 test forms with known words)")
        print(funsd.to_string(index=False))

    checks = group_check()
    if checks is not None:
        checks.to_csv(TABLES / "group_check.csv", index=False)
        print("\nScan group check")
        print(checks.to_string(index=False))

    print(f"\nPer-page table: {args.out}\nSummary tables: {TABLES}")


if __name__ == "__main__":
    main()
