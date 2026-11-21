"""TF-IDF classifier on each text option, cross-validated on the shared folds.

Runs: the four cleaned text options (C, hybrid, both, D), engine A's text, C before Bates
removal, and a majority-class floor. Model: words and two-word phrases (TF-IDF, in at
least 2 pages) and logistic regression with balanced class weights, default settings.
Scored by macro F1 and accuracy per fold. final_test.py imports tfidf_model(),
read_texts() and TEXT_DIRS from here.

Reads results/cv_folds.csv and data/pm_sample/ocr_text_clean/.
Writes results/classifier/tfidf_cv_results.csv, tfidf_cv_summary.csv and
tfidf_cv_predictions.csv.

Usage:
    python pipeline/03_classify/train_tfidf.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import make_pipeline

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "pm_sample"
FOLDS = ROOT / "results" / "cv_folds.csv"
OUT = ROOT / "results" / "classifier"

TEXT_DIRS = {
    "C": DATA / "ocr_text_clean" / "C",
    "hybrid": DATA / "ocr_text_clean" / "hybrid",
    "both": DATA / "ocr_text_clean" / "both",
    "D": DATA / "ocr_text_clean" / "D",
    "C_with_bates": DATA / "ocr_text" / "C",
    "A": DATA / "ocr_text_clean" / "A",
}
SEED = 44


# Models

def tfidf_model():
    """Return the TF-IDF model: word and two-word features, balanced logistic regression."""
    return make_pipeline(
        TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True),
        LogisticRegression(class_weight="balanced", max_iter=2000, random_state=SEED),
    )


def majority_model():
    """Return a baseline that always predicts the largest department."""
    return DummyClassifier(strategy="most_frequent")


# Cross-validation

def read_texts(folder: Path, ids: pd.Series) -> pd.Series:
    """Return the texts of ids from folder (empty when a file is missing)."""
    def read(doc_id):
        path = folder / f"{doc_id}.txt"
        return path.read_text(encoding="utf-8", errors="replace") if path.exists() else ""
    return ids.map(read)


def cross_validate(run: str, texts: pd.Series, folds: pd.DataFrame, make_model):
    """Score one run on the shared folds."""
    results, predictions = [], []
    for fold in sorted(folds["fold"].unique()):
        train = folds["fold"] != fold
        test = ~train
        model = make_model()
        model.fit(texts[train], folds.loc[train, "department"])
        pred = model.predict(texts[test])
        conf = model.predict_proba(texts[test]).max(axis=1)
        true = folds.loc[test, "department"]
        results.append({
            "run": run, "fold": fold, "test_pages": int(test.sum()),
            "macro_f1": f1_score(true, pred, average="macro", zero_division=0),
            "accuracy": accuracy_score(true, pred),
        })
        predictions.append(pd.DataFrame({
            "run": run, "id": folds.loc[test, "id"], "fold": fold,
            "true": true, "predicted": pred, "confidence": conf.round(3),
        }))
    return results, predictions


# Main

def main() -> None:
    """Run every text option and save the results."""
    folds = pd.read_csv(FOLDS, dtype={"id": str}).reset_index(drop=True)
    OUT.mkdir(parents=True, exist_ok=True)

    all_results, all_predictions = [], []
    runs = [(name, folder, tfidf_model) for name, folder in TEXT_DIRS.items()]
    runs.insert(4, ("majority", TEXT_DIRS["C"], majority_model))

    for name, folder, make_model in runs:
        texts = read_texts(folder, folds["id"])
        empty = int((texts.str.strip() == "").sum())
        print(f"{name}: {len(texts)} pages, {empty} empty", flush=True)
        results, predictions = cross_validate(name, texts, folds, make_model)
        all_results += results
        all_predictions += predictions

    results = pd.DataFrame(all_results)
    results.to_csv(OUT / "tfidf_cv_results.csv", index=False)
    pd.concat(all_predictions).to_csv(OUT / "tfidf_cv_predictions.csv", index=False)

    summary = results.groupby("run", sort=False).agg(
        macro_f1_mean=("macro_f1", "mean"), macro_f1_sd=("macro_f1", "std"),
        accuracy_mean=("accuracy", "mean"), accuracy_sd=("accuracy", "std"),
    ).round(3)
    summary.to_csv(OUT / "tfidf_cv_summary.csv")
    print("\n" + summary.to_string())
    print(f"\nSaved to {OUT}")


if __name__ == "__main__":
    main()
