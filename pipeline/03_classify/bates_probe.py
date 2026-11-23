"""Can the department be guessed from the Bates number alone?

Bates numbers were stamped in runs, box by box, so their first digits could carry the
department. A model given only the first 4 digits of each page's Bates number is scored
on the same folds as every other model. If it beats the floor (always guessing
Marketing) by a wide margin, the stamp is a shortcut risk and removing it from the
classifier's text matters. Model: the 4-digit prefix one-hot encoded, logistic
regression with balanced class weights.

Reads results/cv_folds.csv and data/pm_sample/manifest.csv.
Writes results/classifier/bates_probe.csv (one row per fold, plus the mean).

Usage:
    python -m pipeline.03_classify.bates_probe
"""

from __future__ import annotations

import re

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder

from pipeline.common import ROOT

FOLDS = ROOT / "results" / "cv_folds.csv"
MANIFEST = ROOT / "data" / "pm_sample" / "manifest.csv"
OUT = ROOT / "results" / "classifier" / "bates_probe.csv"

PREFIX_DIGITS = 4
SEED = 44


def bates_prefix(bates: str) -> str:
    """Return the first 4 digits of a page's Bates number."""
    digits = re.sub(r"\D", "", str(bates))
    return digits[:PREFIX_DIGITS] if digits else "none"


def main() -> None:
    """Score the prefix-only model on every fold and save the results."""
    folds = pd.read_csv(FOLDS, dtype={"id": str})
    manifest = pd.read_csv(MANIFEST, dtype=str)[["id", "bates"]]
    df = folds.merge(manifest, on="id", how="left").reset_index(drop=True)
    df["prefix"] = df["bates"].map(bates_prefix)

    rows = []
    for fold in sorted(df["fold"].unique()):
        train = df["fold"] != fold
        test = ~train
        model = make_pipeline(
            OneHotEncoder(handle_unknown="ignore"),
            LogisticRegression(class_weight="balanced", max_iter=2000, random_state=SEED),
        )
        model.fit(df.loc[train, ["prefix"]], df.loc[train, "department"])
        pred = model.predict(df.loc[test, ["prefix"]])
        true = df.loc[test, "department"]
        seen = df.loc[test, "prefix"].isin(set(df.loc[train, "prefix"]))
        rows.append({
            "fold": fold, "test_pages": int(test.sum()),
            "prefix_seen_in_training": int(seen.sum()),
            "macro_f1": round(f1_score(true, pred, average="macro", zero_division=0), 3),
            "accuracy": round(accuracy_score(true, pred), 3),
        })

    result = pd.DataFrame(rows)
    mean = result.drop(columns="fold").mean().round(3).to_dict()
    result["fold"] = result["fold"].astype(str)
    result = pd.concat([result, pd.DataFrame([{"fold": "mean", **mean}])], ignore_index=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(OUT, index=False)

    print(f"Different {PREFIX_DIGITS}-digit prefixes: {df['prefix'].nunique()} across {len(df)} pages\n")
    print(result.to_string(index=False))
    print("\nFloor (always Marketing): macro F1 0.069, accuracy 0.260 | TF-IDF hybrid: 0.563, 0.612")
    print(f"Saved: {OUT}")


if __name__ == "__main__":
    main()
