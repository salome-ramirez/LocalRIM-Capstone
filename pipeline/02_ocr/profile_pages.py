"""Measure each page's scan quality from the image and group similar pages.

Measures, on each first page resized to a common width: brightness, contrast, background
(90th percentile gray), dark_share, ink_share, speckle (lone dark pixels), sharpness
(variance of the Laplacian), plus page size and native dpi from pdf_info.csv. Pages are
then grouped with k-means on the standardised measures. No OCR is run.

Reads data/pm_sample/images_native/<id>.png (or images/ when missing) and pdf_info.csv.
Writes data/pm_sample/page_profile.csv (measures and group) and results/page_groups.html
(example pages per group, for naming them).

Usage:
    python pipeline/02_ocr/profile_pages.py              # 4 groups
    python pipeline/02_ocr/profile_pages.py --groups 5
"""

import argparse
import base64
import io
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
PM_DIR = ROOT / "data" / "pm_sample"
NATIVE_DIR = PM_DIR / "images_native"
OLD_DIR = PM_DIR / "images"
MANIFEST = PM_DIR / "manifest.csv"
PDF_INFO = PM_DIR / "pdf_info.csv"
OUT_CSV = PM_DIR / "page_profile.csv"
OUT_HTML = ROOT / "results" / "page_groups.html"

WORK_WIDTH = 1000        # every page is measured at this width, so dpi doesn't skew the numbers
DARK = 80                # gray level below this = very dark
INK = 160                # gray level below this = ink
SEED = 44
EXAMPLES_PER_GROUP = 12
NOT_RETRIEVED = {"qnkl0085", "hzgp0030", "xpxk0085"}   # never downloaded
FEATURES = ["brightness", "contrast", "background", "dark_share", "ink_share", "speckle", "sharpness"]


def measure(path: Path) -> dict:
    """Return the scan-quality measures of one page image."""
    img = Image.open(path).convert("L")
    if img.width != WORK_WIDTH:
        img = img.resize((WORK_WIDTH, round(img.height * WORK_WIDTH / img.width)), Image.LANCZOS)
    g = np.asarray(img, dtype=np.float32)

    dark = g < DARK
    ink = g < INK
    # Speckle: a dark pixel with no dark neighbours in its 3x3 square is a lone dot (scan noise).
    padded = np.pad(dark, 1)
    neighbours = sum(np.roll(np.roll(padded, dy, 0), dx, 1)
                     for dy in (-1, 0, 1) for dx in (-1, 0, 1) if (dy, dx) != (0, 0))[1:-1, 1:-1]
    lone = dark & (neighbours == 0)
    # Sharpness: variance of the Laplacian (how strong the edges are).
    lap = (-4 * g[1:-1, 1:-1] + g[:-2, 1:-1] + g[2:, 1:-1] + g[1:-1, :-2] + g[1:-1, 2:])

    return {
        "brightness": round(float(g.mean()), 1),
        "contrast": round(float(g.std()), 1),
        "background": round(float(np.percentile(g, 90)), 1),
        "dark_share": round(float(dark.mean()), 4),
        "ink_share": round(float(ink.mean()), 4),
        "speckle": round(float(lone.sum() / max(dark.sum(), 1)), 4),
        "sharpness": round(float(lap.var()), 1),
    }


def thumb(path: Path, width=180) -> str:
    """Return a small grayscale base64 PNG of a page for the example sheet."""
    img = Image.open(path).convert("L")
    img.thumbnail((width, width * 2))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def main():
    """Measure every page, group them and write the profile and example sheet."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--groups", type=int, default=4)
    args = parser.parse_args()

    try:
        from sklearn.cluster import KMeans
        from sklearn.preprocessing import StandardScaler
    except ImportError:
        print("scikit-learn is missing: pip install scikit-learn")
        return

    manifest = pd.read_csv(MANIFEST, dtype=str)
    manifest["position"] = range(1, len(manifest) + 1)
    info = pd.read_csv(PDF_INFO, dtype=str).drop_duplicates("id", keep="last") if PDF_INFO.exists() else None

    rows = []
    for doc_id, pos in zip(manifest["id"], manifest["position"]):
        if doc_id in NOT_RETRIEVED:
            continue
        path = NATIVE_DIR / f"{doc_id}.png"
        source = "native"
        if not path.exists():
            path, source = OLD_DIR / f"{doc_id}.png", "150dpi"
        if not path.exists():
            continue
        row = {"id": doc_id, "position": pos, "source": source, **measure(path)}
        rows.append(row)
        if len(rows) % 100 == 0:
            print(f"  measured {len(rows)} pages", flush=True)

    df = pd.DataFrame(rows)
    if info is not None:
        info = info.astype({"render_dpi": float, "width_px": float, "height_px": float})
        info["page_w_in"] = (info["width_px"] / info["render_dpi"]).round(1)
        info["page_h_in"] = (info["height_px"] / info["render_dpi"]).round(1)
        df = df.merge(info[["id", "native_dpi", "page_w_in", "page_h_in"]], on="id", how="left")

    # Group pages by their measures (standardized so no single measure dominates)
    X = StandardScaler().fit_transform(df[FEATURES])
    km = KMeans(n_clusters=args.groups, random_state=SEED, n_init=10).fit(X)
    df["group"] = km.labels_ + 1
    df.to_csv(OUT_CSV, index=False)

    # What each group looks like on average, plus example pages to name it
    summary = df.groupby("group")[FEATURES].median().round(3)
    summary.insert(0, "pages", df["group"].value_counts().sort_index())
    print("\nGroups (median of each measure):")
    print(summary.to_string())

    html = ["<html><head><meta charset='utf-8'><title>Page groups</title><style>",
            "body{font-family:sans-serif;margin:24px;background:#fafafa}",
            "h2{margin-top:32px}table{border-collapse:collapse;font-size:13px}",
            "td,th{border:1px solid #ddd;padding:4px 8px}",
            ".grid{display:flex;flex-wrap:wrap;gap:10px}",
            ".card{background:#fff;border:1px solid #ddd;padding:6px;font-size:11px;width:180px}",
            "</style></head><body><h1>Page groups by scan quality</h1>",
            "<p>Each group was found from image measures alone. Look at the examples and give each group a name.</p>",
            summary.to_html()]
    for g in sorted(df["group"].unique()):
        part = df[df["group"] == g]
        html.append(f"<h2>Group {g} ({len(part)} pages)</h2><div class='grid'>")
        for _, r in part.sample(min(EXAMPLES_PER_GROUP, len(part)), random_state=SEED).iterrows():
            p = (NATIVE_DIR if r["source"] == "native" else OLD_DIR) / f"{r['id']}.png"
            html.append(f"<div class='card'><img src='data:image/png;base64,{thumb(p)}' width='180'><br>"
                        f"#{r['position']} · contrast {r['contrast']} · speckle {r['speckle']}</div>")
        html.append("</div>")
    html.append("</body></html>")
    OUT_HTML.parent.mkdir(exist_ok=True)
    OUT_HTML.write_text("\n".join(html), encoding="utf-8")

    n_native = int((df["source"] == "native").sum())
    print(f"\n{len(df)} pages measured ({n_native} from the original scans, {len(df) - n_native} from 150-dpi copies).")
    print(f"Saved {OUT_CSV.relative_to(ROOT)} and {OUT_HTML.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
