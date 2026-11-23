"""Remove each page's own Bates number from the text the classifier sees.

The Bates number is the record's file number and stays with the record, but the
classifier must not see it: the numbers were stamped box by box, so a model could learn
number ranges instead of reading the page. remove_own_bates() drops the page's own
Bates, including the garbled, upside-down and mirrored versions OCR produces, and keeps
other numbers (dates, phone numbers, amounts, ZIP+4 codes).

Reads data/pm_sample/ocr_text/<option>/ (and engine A's text in data/pm_sample/ocr/).
Writes data/pm_sample/ocr_text_clean/<option>/<id>.txt and results/bates_removals.csv.

Usage:
    python -m pipeline.03_classify.prep_classifier_text
"""

from __future__ import annotations

import re

import pandas as pd

from pipeline.common import ROOT
from ocr_eda import NOT_RETRIEVED, is_bates_like, matches_bates

DATA = ROOT / "data" / "pm_sample"
TEXT_IN = DATA / "ocr_text"
TEXT_OUT = DATA / "ocr_text_clean"
LOG = ROOT / "results" / "bates_removals.csv"

OPTIONS = ["C", "hybrid", "both", "D", "A"]
SOURCE_DIR = {"A": DATA / "ocr"}   # every other option is read from TEXT_IN/<option>

# ZIP+4 codes (e.g. 10017-5592) look like a garbled Bates but are address content: never removed
ZIP_PLUS_4 = re.compile(r"^\d{5}-\d{4}$")


# Removal

def remove_own_bates(text: str, bates: str) -> tuple[str, list[str]]:
    """Return the text without tokens that read as this page's Bates, and those tokens."""
    removed = []
    clean_lines = []
    for line in text.splitlines():
        kept = []
        for token in line.split():
            if ZIP_PLUS_4.match(token.strip(".,;:()'\"")):
                kept.append(token)
            elif is_bates_like(token) and matches_bates(token, bates):
                removed.append(token)
            else:
                kept.append(token)
        clean_lines.append(" ".join(kept))
    return "\n".join(clean_lines), removed


# Main

def main() -> None:
    """Clean every text option and log what was removed."""
    manifest = pd.read_csv(DATA / "manifest.csv", dtype=str)
    manifest = manifest[~manifest["id"].isin(NOT_RETRIEVED)]
    bates_of = dict(zip(manifest["id"], manifest["bates"].fillna("")))

    rows = []
    for option in OPTIONS:
        (TEXT_OUT / option).mkdir(parents=True, exist_ok=True)
        for doc_id, bates in bates_of.items():
            path = SOURCE_DIR.get(option, TEXT_IN / option) / f"{doc_id}.txt"
            if not path.exists():
                rows.append({"id": doc_id, "option": option, "bates": bates,
                             "has_text": False, "removed_count": 0, "removed_tokens": ""})
                continue
            text = path.read_text(encoding="utf-8", errors="replace")
            clean, removed = remove_own_bates(text, bates)
            (TEXT_OUT / option / f"{doc_id}.txt").write_text(clean, encoding="utf-8")
            rows.append({"id": doc_id, "option": option, "bates": bates,
                         "has_text": True, "removed_count": len(removed),
                         "removed_tokens": " | ".join(removed)})

    log = pd.DataFrame(rows)
    LOG.parent.mkdir(parents=True, exist_ok=True)
    log.to_csv(LOG, index=False)

    # Summary: pages per option where the page's own Bates was found and removed
    summary = log.groupby("option").agg(
        pages=("id", "count"),
        pages_with_text=("has_text", "sum"),
        pages_bates_removed=("removed_count", lambda s: int((s > 0).sum())),
        tokens_removed=("removed_count", "sum"),
    ).reindex(OPTIONS)
    print(summary.to_string())
    print(f"\nClean texts: {TEXT_OUT}")
    print(f"Removal log: {LOG}")


if __name__ == "__main__":
    main()
