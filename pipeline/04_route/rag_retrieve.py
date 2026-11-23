"""Retrieval check: do the most similar filed pages point a reviewer to the right department?

For every training page, the five most similar pages by meaning (cosine similarity of
the page embeddings) are found in the other folds only, so a page never finds itself or a
page scored alongside it. The neighbours vote for a department, weighted by similarity.
On the review queue (pages the routing sent to a person) this is compared with the
classifier's first and second choice, from the same calibrated model, folds and seed as
train_embeddings.py. The rule, set before results: retrieval helps if its top suggestion
is right more often than the classifier's first choice on the review queue.

Reads results/cv_folds.csv, results/embeddings/ and results/classifier/emb_cv_predictions.csv.
Writes results/rag/neighbors.csv (rank 1-5 per page), retrieval_per_page.csv and
retrieval_summary.csv.

Usage:
    python pipeline/04_route/rag_retrieve.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import sys
sys.path[:0] = [str(Path(__file__).resolve().parents[2] / "pipeline" / step) for step in ("03_classify",)]  # sibling pipeline steps
from train_embeddings import calibrated_model, load_vectors

ROOT = Path(__file__).resolve().parents[2]
FOLDS = ROOT / "results" / "cv_folds.csv"
ROUTED = ROOT / "results" / "classifier" / "emb_cv_predictions.csv"
OUT = ROOT / "results" / "rag"

K = 5


def top2(scores: dict) -> list[str]:
    """Return the two departments with the highest weighted votes."""
    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    return [k for k, _ in ranked[:2]] + [""] * (2 - len(ranked[:2]))


def main() -> None:
    """Find the neighbours of every training page, score the vote and save it."""
    folds = pd.read_csv(FOLDS, dtype={"id": str}).reset_index(drop=True)
    X = load_vectors(folds["id"], "hybrid")
    y = folds["department"].to_numpy()
    ids = folds["id"].to_numpy()
    routed = pd.read_csv(ROUTED, dtype={"id": str}).set_index("id")["routed"]

    neighbor_rows, page_rows = [], []
    for fold in sorted(folds["fold"].unique()):
        train = (folds["fold"] != fold).to_numpy()
        test = ~train

        # Classifier's first and second choices (same calibrated model as routing)
        model = calibrated_model().fit(X[train], y[train])
        proba = model.predict_proba(X[test])
        order = np.argsort(-proba, axis=1)
        cls_first = model.classes_[order[:, 0]]
        cls_second = model.classes_[order[:, 1]]

        # Nearest training pages by meaning (vectors are normalized: dot = cosine)
        sims = X[test] @ X[train].T
        train_ids, train_y = ids[train], y[train]
        for row, i in enumerate(np.where(test)[0]):
            nn = np.argsort(-sims[row])[:K]
            votes = {}
            for rank, j in enumerate(nn, 1):
                votes[train_y[j]] = votes.get(train_y[j], 0.0) + float(sims[row, j])
                neighbor_rows.append({"id": ids[i], "fold": fold, "rank": rank, "neighbor_id": train_ids[j],
                                      "neighbor_department": train_y[j], "similarity": round(float(sims[row, j]), 4)})
            nb1, nb2 = top2(votes)
            page_rows.append({
                "id": ids[i], "fold": fold, "true": y[i], "routed": routed.get(ids[i], ""),
                "neighbors_first": nb1, "neighbors_second": nb2,
                "classifier_first": cls_first[row], "classifier_second": cls_second[row],
                "top_similarity": round(float(sims[row, nn[0]]), 4),
            })

    pages = pd.DataFrame(page_rows)
    for who in ("neighbors", "classifier"):
        pages[f"{who}_top1"] = pages[f"{who}_first"] == pages["true"]
        pages[f"{who}_top2"] = pages[f"{who}_top1"] | (pages[f"{who}_second"] == pages["true"])

    def summarize(df, name):
        return {"pages": name, "n": len(df),
                "classifier_top1": round(df["classifier_top1"].mean(), 3),
                "neighbors_top1": round(df["neighbors_top1"].mean(), 3),
                "classifier_top2": round(df["classifier_top2"].mean(), 3),
                "neighbors_top2": round(df["neighbors_top2"].mean(), 3)}

    review = pages[pages["routed"] == "review"]
    summary = pd.DataFrame([summarize(review, "review queue"), summarize(pages, "all pages")])

    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(neighbor_rows).to_csv(OUT / "neighbors.csv", index=False)
    pages.to_csv(OUT / "retrieval_per_page.csv", index=False)
    summary.to_csv(OUT / "retrieval_summary.csv", index=False)

    print(summary.to_string(index=False))
    print("\nSuccess rule (set before results): neighbors_top1 > classifier_top1 on the review queue")
    print(f"Saved to {OUT}")


if __name__ == "__main__":
    main()
