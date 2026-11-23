"""Department classifier on page embeddings, cross-validated on the shared folds.

emb_model() is a logistic regression with balanced class weights on standardised
vectors; calibrated_model() adds sigmoid calibration with 3 inner folds. The routing
threshold is chosen on training folds only, with review_routing.py's rules.
load_vectors() reads the saved embeddings by record ID. The app's pipeline uses all
three.

--text picks the embeddings: hybrid (default), hybrid_with_bates or A.

Reads results/cv_folds.csv, results/embeddings/ and results/classifier/tfidf_cv_summary.csv.
Writes results/classifier/emb[_<text>]_cv_results.csv, _cv_predictions.csv,
_routing_curve.csv and _calibration_summary.csv.

Usage:
    python -m pipeline.03_classify.train_embeddings
    python -m pipeline.03_classify.train_embeddings --text hybrid_with_bates
    python -m pipeline.03_classify.train_embeddings --text A
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, log_loss
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from pipeline.common import ROOT
from review_routing import (SEED, TARGET_ACCURACY, THRESHOLDS, choose_threshold,
                                     expected_calibration_error, routing_at)

FOLDS = ROOT / "results" / "cv_folds.csv"
EMB_DIR = ROOT / "results" / "embeddings"
MODEL_TAG = "qwen3-0.6b"
OUT = ROOT / "results" / "classifier"


def emb_model():
    """Return the classifier: standardised vectors, logistic regression, balanced class weights."""
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(class_weight="balanced", max_iter=5000, random_state=SEED),
    )


def calibrated_model():
    """Return emb_model() with sigmoid calibration (3 inner folds)."""
    inner = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)
    return CalibratedClassifierCV(emb_model(), method="sigmoid", cv=inner)


def load_vectors(ids: pd.Series, text_name: str) -> np.ndarray:
    """Return the saved embeddings of the given IDs, in that order."""
    run_name = f"{text_name}_{MODEL_TAG}"
    vectors = np.load(EMB_DIR / f"{run_name}.npy")
    order = pd.read_csv(EMB_DIR / f"{run_name}_ids.csv", dtype={"id": str})["id"]
    row_of = {doc_id: n for n, doc_id in enumerate(order)}
    return vectors[[row_of[i] for i in ids]]


def main() -> None:
    """Run the cross-validated evaluation for one text option."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", choices=["hybrid", "hybrid_with_bates", "A"], default="hybrid")
    text_name = parser.parse_args().text
    prefix = "emb" if text_name == "hybrid" else f"emb_{text_name}"

    folds = pd.read_csv(FOLDS, dtype={"id": str}).reset_index(drop=True)
    X = load_vectors(folds["id"], text_name)
    y = folds["department"].to_numpy()

    rows, pages, raw_list, cal_list = [], [], [], []
    for fold in sorted(folds["fold"].unique()):
        train = (folds["fold"] != fold).to_numpy()
        test = ~train

        raw = emb_model().fit(X[train], y[train])
        raw_proba = raw.predict_proba(X[test])
        raw_pred = raw.classes_[raw_proba.argmax(axis=1)]

        cal = calibrated_model().fit(X[train], y[train])
        cal_proba = cal.predict_proba(X[test])
        cal_pred = cal.classes_[cal_proba.argmax(axis=1)]
        cal_conf = cal_proba.max(axis=1)

        inner = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
        train_proba = cross_val_predict(calibrated_model(), X[train], y[train],
                                        cv=inner, method="predict_proba")
        train_correct = np.sort(np.unique(y[train]))[train_proba.argmax(axis=1)] == y[train]
        threshold = choose_threshold(train_proba.max(axis=1), train_correct)

        correct = cal_pred == y[test]
        routed = routing_at(cal_conf, correct, threshold)
        rows.append({
            "fold": fold, "test_pages": int(test.sum()),
            "macro_f1": f1_score(y[test], raw_pred, average="macro", zero_division=0),
            "accuracy": accuracy_score(y[test], raw_pred),
            "threshold_from_training": threshold,
            "auto_filed": routed["auto_filed"], "coverage": routed["coverage"],
            "accuracy_auto_filed": routed["accuracy_auto_filed"],
        })
        pages.append(pd.DataFrame({
            "id": folds.loc[test, "id"].to_numpy(), "fold": fold, "true": y[test],
            "predicted_raw": raw_pred, "confidence_raw": raw_proba.max(axis=1).round(3),
            "predicted": cal_pred, "confidence": cal_conf.round(3), "correct": correct,
            "routed": np.where(cal_conf >= threshold, "auto-filed", "review"),
        }))
        raw_list.append(raw_proba)
        cal_list.append(cal_proba)

    results = pd.DataFrame(rows)
    pages = pd.concat(pages, ignore_index=True)
    classes = np.sort(np.unique(y))
    raw_correct = (pages["predicted_raw"] == pages["true"]).to_numpy()
    cal_correct = pages["correct"].to_numpy()

    calibration = pd.DataFrame([
        {"model": "raw", "accuracy": round(raw_correct.mean(), 3),
         "ece": expected_calibration_error(pages["confidence_raw"].to_numpy(), raw_correct),
         "log_loss": round(log_loss(pages["true"], np.vstack(raw_list), labels=classes), 3)},
        {"model": "calibrated", "accuracy": round(cal_correct.mean(), 3),
         "ece": expected_calibration_error(pages["confidence"].to_numpy(), cal_correct),
         "log_loss": round(log_loss(pages["true"], np.vstack(cal_list), labels=classes), 3)},
    ])
    curve = pd.DataFrame([routing_at(pages["confidence"].to_numpy(), cal_correct, t) for t in THRESHOLDS])

    OUT.mkdir(parents=True, exist_ok=True)
    results.round(3).to_csv(OUT / f"{prefix}_cv_results.csv", index=False)
    pages.to_csv(OUT / f"{prefix}_cv_predictions.csv", index=False)
    curve.to_csv(OUT / f"{prefix}_routing_curve.csv", index=False)
    calibration.to_csv(OUT / f"{prefix}_calibration_summary.csv", index=False)

    tfidf = pd.read_csv(OUT / "tfidf_cv_summary.csv", index_col="run")
    print("Comparison (mean ± spread over 5 folds):")
    print(f"  TF-IDF hybrid     macro F1 {tfidf.loc['hybrid', 'macro_f1_mean']:.3f} ± {tfidf.loc['hybrid', 'macro_f1_sd']:.3f}"
          f" | accuracy {tfidf.loc['hybrid', 'accuracy_mean']:.3f}")
    if text_name != "hybrid" and (OUT / "emb_cv_results.csv").exists():
        base = pd.read_csv(OUT / "emb_cv_results.csv")
        print(f"  Embeddings hybrid macro F1 {base['macro_f1'].mean():.3f} ± {base['macro_f1'].std():.3f}"
              f" | accuracy {base['accuracy'].mean():.3f}")
    print(f"  Embeddings {text_name:<7}macro F1 {results['macro_f1'].mean():.3f} ± {results['macro_f1'].std():.3f}"
          f" | accuracy {results['accuracy'].mean():.3f}")
    print("\nCalibration:")
    print(calibration.to_string(index=False))
    print("\nRouting curve, calibrated:")
    print(curve.to_string(index=False))
    auto = pages["routed"] == "auto-filed"
    print(f"\nTarget {TARGET_ACCURACY:.0%}, threshold from training folds, per fold:")
    print(results[["fold", "threshold_from_training", "auto_filed", "coverage", "accuracy_auto_filed"]].to_string(index=False))
    acc_auto = pages.loc[auto, "correct"].mean() if auto.any() else float("nan")
    print(f"\nOverall: {auto.sum()} of {len(pages)} auto-filed ({auto.mean():.0%}), accuracy {acc_auto:.1%}")
    print(f"Saved to {OUT}")


if __name__ == "__main__":
    main()
