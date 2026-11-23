"""Name the scan-quality groups and split k-means group 3 in two.

Group 3 held two different problems, degraded scans and printed graphics, which the
speckle measure separates; it is split by a fixed rule (speckle > 0.005 -> 3a, otherwise
3b) instead of re-clustering, which would reshuffle every group.

Final groups:
    1   Clean typed files              letters, memos, correspondence
    2   Photographic ads               grayscale photographic ads on dark backgrounds
    3a  Degraded scans                 streaky or blotchy copies, maps, folder covers
    3b  Graphics and clippings         high-contrast ads, clippings, flyers
    4   Clean reports, forms, tables   dense typed pages

Updates data/pm_sample/page_profile.csv: cluster (k-means number), group, group_name.

Usage (after profile_pages.py):
    python -m pipeline.02_ocr.name_page_groups
"""

from __future__ import annotations


import pandas as pd

from pipeline.common import ROOT

PROFILE = ROOT / "data" / "pm_sample" / "page_profile.csv"

SPLIT_CLUSTER = 3
SPECKLE_SPLIT = 0.005

NAMES = {
    "1": "Clean typed files",
    "2": "Photographic ads",
    "3a": "Degraded scans",
    "3b": "Graphics and clippings",
    "4": "Clean reports, forms, tables",
}


def final_group(row: pd.Series) -> str:
    """Return a page's final group label from its k-means cluster and speckle."""
    if int(row["cluster"]) == SPLIT_CLUSTER:
        return "3a" if row["speckle"] > SPECKLE_SPLIT else "3b"
    return str(int(row["cluster"]))


def main() -> None:
    """Add the final group and its name to every page in the profile."""
    df = pd.read_csv(PROFILE, dtype={"id": str})
    if "cluster" not in df.columns:          # first run: keep the k-means number
        df = df.rename(columns={"group": "cluster"})
    df["group"] = df.apply(final_group, axis=1)
    df["group_name"] = df["group"].map(NAMES)
    df.to_csv(PROFILE, index=False)

    counts = df.groupby(["group", "group_name"]).size().rename("pages").reset_index()
    print(counts.to_string(index=False))
    print(f"\nUpdated {PROFILE.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
