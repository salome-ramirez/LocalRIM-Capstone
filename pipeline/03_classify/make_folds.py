"""Split the 250 training labels into 5 cross-validation folds, once, for every model.

Every classifier is scored on the same folds. The split is stratified (each fold gets a
similar mix of departments) and grouped by archive box (pages from one box always land
in the same fold, so a model is never tested on a near-copy of a page it trained on;
pages with no box are their own group). Departments are scored with Administration
merged into Legal. The final-test labels are not used.

Reads labels/label_pm_labels_<annotator>.csv and data/pm_sample/metadata_full.csv (box).
Writes results/cv_folds.csv (id, department, box, group, fold).

Usage:
    python pipeline/03_classify/make_folds.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

ROOT = Path(__file__).resolve().parents[2]
LABELS = ROOT / "labels" / "label_pm_labels_salo.csv"
METADATA = ROOT / "data" / "pm_sample" / "metadata_full.csv"
OUT = ROOT / "results" / "cv_folds.csv"

N_FOLDS = 5
SEED = 44
MERGE_FOR_SCORING = {"ADM": "LEG"}   # Administration is scored with Legal


def load_training_labels() -> pd.DataFrame:
    """Return the valid training labels, one row per document, with the scored department."""
    labels = pd.read_csv(LABELS, dtype=str)
    labels = labels.sort_values("saved_at").drop_duplicates("doc_id", keep="last")
    labels = labels[labels["outcome"] == "labeled"].copy()
    labels["department"] = labels["function_id"].replace(MERGE_FOR_SCORING)
    return labels[["doc_id", "department"]].rename(columns={"doc_id": "id"})


def main() -> None:
    """Assign and save the folds, then check them."""
    df = load_training_labels()
    meta = pd.read_csv(METADATA, dtype=str, usecols=["id", "box"])
    df = df.merge(meta, on="id", how="left")
    df["box"] = df["box"].fillna("")
    df["group"] = df["box"].where(df["box"] != "", "nobox_" + df["id"])

    splitter = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    df["fold"] = -1
    for fold, (_, test_idx) in enumerate(splitter.split(df, df["department"], df["group"]), 1):
        df.loc[df.index[test_idx], "fold"] = fold

    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)

    # Checks
    print(f"Training pages: {len(df)} | groups: {df['group'].nunique()}\n")
    print("Departments per fold:")
    print(pd.crosstab(df["department"], df["fold"], margins=True).to_string())
    boxes_in_two_folds = df[df["box"] != ""].groupby("box")["fold"].nunique().gt(1).sum()
    print(f"\nBoxes split across folds: {boxes_in_two_folds} (should be 0)")
    print(f"Saved: {OUT}")


if __name__ == "__main__":
    main()
