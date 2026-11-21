"""Extend the OCR test pages so every scan-quality group is represented.

The first 10 test pages were chosen by confidence band before the scan-quality groups
existed, so they lean towards clean pages. The original 10 are kept and pages are added
at random (seed 44) until every group has at least MIN_PER_GROUP test pages.

Reads data/pm_sample/page_profile.csv and results/ocr_spotcheck_pm_pages.csv.
Writes results/ocr_spotcheck_pm_pages_original10.csv (copy, first run only) and the
extended results/ocr_spotcheck_pm_pages.csv, read by ocr_review.py.

Usage:
    python pipeline/02_ocr/extend_test_pages.py
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
PROFILE = ROOT / "data" / "pm_sample" / "page_profile.csv"
PAGES = ROOT / "results" / "ocr_spotcheck_pm_pages.csv"
BACKUP = ROOT / "results" / "ocr_spotcheck_pm_pages_original10.csv"

MIN_PER_GROUP = 4
SEED = 44


def main() -> None:
    """Add random pages until every group has enough, and save the list."""
    if not BACKUP.exists():
        shutil.copy(PAGES, BACKUP)
    original = pd.read_csv(BACKUP, dtype={"id": str})
    profile = pd.read_csv(PROFILE, dtype={"id": str, "group": str})

    pages = original.merge(profile[["id", "group", "group_name"]], on="id", how="left")
    pages["why_picked"] = pages["why_picked"].fillna("Original test page")

    added = []
    for group, pool in profile.groupby("group"):
        have = int((pages["group"] == group).sum())
        need = max(MIN_PER_GROUP - have, 0)
        candidates = pool[~pool["id"].isin(pages["id"])]
        pick = candidates.sample(min(need, len(candidates)), random_state=SEED)
        for _, r in pick.iterrows():
            added.append({"id": r["id"], "position": r["position"],
                          "why_picked": f"Added for group {group} ({r['group_name']})",
                          "group": group, "group_name": r["group_name"]})

    out = pd.concat([pages, pd.DataFrame(added)], ignore_index=True)
    out.to_csv(PAGES, index=False)

    print(out.groupby(["group", "group_name"]).size().rename("test pages").reset_index().to_string(index=False))
    print(f"\n{len(original)} original + {len(added)} added = {len(out)} test pages")
    print(f"Written to {PAGES.relative_to(ROOT)} (original list backed up as {BACKUP.name})")


if __name__ == "__main__":
    main()
