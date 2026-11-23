"""Where a classifier goes wrong, from its saved cross-validated predictions.

For a TF-IDF run (default: hybrid) or, with --embeddings, the embedding model's
uncalibrated predictions: the confusion matrix (rows = label, columns = guess),
precision, recall and F1 per department, and accuracy on pages the labeler marked sure
vs unsure.

Reads results/classifier/tfidf_cv_predictions.csv (or emb_cv_predictions.csv) and the
labels' certainty. Writes results/classifier/confusion_<run>.csv and
per_department_<run>.csv.

Usage:
    python -m pipeline.03_classify.tfidf_errors
    python -m pipeline.03_classify.tfidf_errors --run C
    python -m pipeline.03_classify.tfidf_errors --embeddings
"""

from __future__ import annotations

import argparse

import pandas as pd
from sklearn.metrics import classification_report

from pipeline.common import ROOT, TRAIN_LABELS as LABELS

OUT = ROOT / "results" / "classifier"
PREDICTIONS = OUT / "tfidf_cv_predictions.csv"
EMB_PREDICTIONS = OUT / "emb_cv_predictions.csv"


def main() -> None:
    """Write the confusion matrix and per-department scores of one run."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", default="hybrid")
    parser.add_argument("--embeddings", action="store_true", help="report on the embedding model")
    args = parser.parse_args()

    if args.embeddings:
        run = "embeddings"
        pred = pd.read_csv(EMB_PREDICTIONS, dtype={"id": str})
        pred = pred.drop(columns=["predicted", "correct"]).rename(columns={"predicted_raw": "predicted"})
    else:
        run = args.run
        pred = pd.read_csv(PREDICTIONS, dtype={"id": str})
        pred = pred[pred["run"] == run].copy()

    labels = pd.read_csv(LABELS, dtype=str)
    labels = labels.sort_values("saved_at").drop_duplicates("doc_id", keep="last")
    pred = pred.merge(labels[["doc_id", "certainty"]].rename(columns={"doc_id": "id"}),
                      on="id", how="left")
    pred["correct"] = pred["true"] == pred["predicted"]

    # Confusion matrix
    confusion = pd.crosstab(pred["true"], pred["predicted"],
                            rownames=["label"], colnames=["guess"], margins=True)
    confusion.to_csv(OUT / f"confusion_{run}.csv")

    # Per department
    report = classification_report(pred["true"], pred["predicted"],
                                   output_dict=True, zero_division=0)
    per_dept = (pd.DataFrame(report).T
                .drop(index=["accuracy", "macro avg", "weighted avg"])
                .rename(columns={"support": "pages"})
                .round(3))
    per_dept["pages"] = per_dept["pages"].astype(int)
    per_dept = per_dept.sort_values("f1-score", ascending=False)
    per_dept.to_csv(OUT / f"per_department_{run}.csv")

    # Sure vs unsure
    by_certainty = pred.groupby("certainty")["correct"].agg(pages="count", accuracy="mean").round(3)

    print(f"Run: {run} | {len(pred)} pages\n")
    print("Confusion matrix (rows = label, columns = model's guess):")
    print(confusion.to_string())
    print("\nPer department:")
    print(per_dept.to_string())
    print("\nAccuracy by Salo's certainty:")
    print(by_certainty.to_string())
    print(f"\nSaved to {OUT}")


if __name__ == "__main__":
    main()
