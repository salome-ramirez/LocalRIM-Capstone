"""Build the four candidate texts the department classifier is compared on.

PaddleOCR (engine C) is the main engine. Which text the classifier reads is decided by
cross-validated filing accuracy over four options:
    C        PaddleOCR on every page
    hybrid   GLM-OCR (engine D) on the hard scan groups (2, 3a, 3b), PaddleOCR elsewhere
    both     PaddleOCR and GLM-OCR text joined
    D        GLM-OCR on every page
Where GLM-OCR has no reading (2 pages crashed the model), PaddleOCR's text is used and
the fallback is recorded. The page's own Bates number is removed later
(prep_classifier_text.py). No OCR is run.

Reads data/pm_sample/manifest.csv, page_profile.csv, ocr_paddle/ and ocr_vlm/.
Writes data/pm_sample/ocr_text/<option>/<id>.txt and data/pm_sample/ocr_text/sources.csv
(per page: group and the engine each option used).

Usage:
    python -m pipeline.02_ocr.build_ocr_texts
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from pipeline.common import ROOT

DATA = ROOT / "data" / "pm_sample"
ENGINE_C = DATA / "ocr_paddle"
ENGINE_D = DATA / "ocr_vlm"
OUT = DATA / "ocr_text"

NOT_RETRIEVED = {"qnkl0085", "hzgp0030", "xpxk0085"}   # never downloaded
HARD_GROUPS = {"2", "3a", "3b"}                          # the four text options
OPTIONS = ["C", "hybrid", "both", "D"]


def read_text(folder: Path, doc_id: str) -> str | None:
    """Return an engine's text for a page, or None when it has no usable reading."""
    path = folder / f"{doc_id}.txt"
    if not path.exists():
        return None
    text = path.read_text(encoding="utf-8", errors="replace").strip()
    return text or None


def main() -> None:
    """Write the four text options and the sources table."""
    manifest = pd.read_csv(DATA / "manifest.csv", dtype=str)
    ids = [i for i in manifest["id"] if i not in NOT_RETRIEVED]
    groups = pd.read_csv(DATA / "page_profile.csv", dtype={"id": str, "group": str})
    group_of = dict(zip(groups["id"], groups["group"]))

    for option in OPTIONS:
        (OUT / option).mkdir(parents=True, exist_ok=True)

    rows = []
    for doc_id in ids:
        c = read_text(ENGINE_C, doc_id) or ""
        d = read_text(ENGINE_D, doc_id)
        group = group_of.get(doc_id, "")
        d_or_c = d if d is not None else c
        hard = group in HARD_GROUPS

        texts = {
            "C": c,
            "hybrid": d_or_c if hard else c,
            "both": "\n".join(t for t in (c, d) if t),
            "D": d_or_c,
        }
        for option, text in texts.items():
            (OUT / option / f"{doc_id}.txt").write_text(text, encoding="utf-8")

        rows.append({
            "id": doc_id,
            "group": group,
            "d_available": d is not None,
            "hybrid_source": ("D" if d is not None else "C (fallback)") if hard else "C",
            "D_source": "D" if d is not None else "C (fallback)",
        })

    log = pd.DataFrame(rows)
    log.to_csv(OUT / "sources.csv", index=False)

    print(f"Pages: {len(log)}")
    print(f"Hard-group pages (2, 3a, 3b): {(log['group'].isin(HARD_GROUPS)).sum()}")
    print("Hybrid sources:")
    print(log["hybrid_source"].value_counts().to_string())
    print(f"Pages without a GLM-OCR reading (C used instead): {(~log['d_available']).sum()}")
    missing = log.loc[log["group"] == "", "id"].tolist()
    if missing:
        print(f"Pages with no scan group (treated as not hard): {len(missing)}")
    print(f"Written to {OUT}")


if __name__ == "__main__":
    main()
