"""Calibration and review routing, and the routing rules the classifiers share.

Confidence is calibrated inside each training fold (sigmoid). The routing curve gives,
for thresholds 0.30-0.90, the share of pages filed automatically and their accuracy. The
threshold is the lowest one at which the auto-filed training pages reach 90% accuracy;
it is then applied unchanged to the held-out fold. main() runs this for the TF-IDF model;
train_embeddings.py imports the same functions.

Reads results/cv_folds.csv and data/pm_sample/ocr_text_clean/hybrid/.
Writes results/classifier/routing_curve.csv, routing_by_fold.csv,
calibration_summary.csv and routing_predictions.csv.

Usage:
    python -m pipeline.04_route.review_routing
"""

from __future__ import annotations


import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline

from pipeline.common import ROOT

FOLDS = ROOT / "results" / "cv_folds.csv"
TEXTS = ROOT / "data" / "pm_sample" / "ocr_text_clean" / "hybrid"
OUT = ROOT / "results" / "classifier"

TARGET_ACCURACY = 0.90
THRESHOLDS = np.round(np.arange(0.30, 0.91, 0.05), 2)
MIN_PAGES_FOR_THRESHOLD = 10    # a threshold must auto-file at least this many training pages
ECE_BINS = 10
SEED = 44


# Models

def tfidf_model():
    """Same settings as train_tfidf.py."""
    return make_pipeline(
        TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True),
        LogisticRegression(class_weight="balanced", max_iter=2000, random_state=SEED),
    )


def calibrated_model():
    """Return the TF-IDF model with sigmoid calibration (3 inner folds)."""
    inner = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)
    return CalibratedClassifierCV(tfidf_model(), method="sigmoid", cv=inner)


# Measures

def expected_calibration_error(confidence, correct, bins=ECE_BINS) -> float:
    """Average gap between stated confidence and actual accuracy, weighted by bin size."""
    edges = np.linspace(0, 1, bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        in_bin = (confidence > lo) & (confidence <= hi)
        if in_bin.any():
            ece += in_bin.mean() * abs(correct[in_bin].mean() - confidence[in_bin].mean())
    return round(float(ece), 3)


def routing_at(confidence, correct, threshold):
    """Return coverage and accuracy of the pages auto-filed at a threshold."""
    auto = confidence >= threshold
    return {
        "threshold": threshold,
        "auto_filed": int(auto.sum()),
        "coverage": round(float(auto.mean()), 3),
        "accuracy_auto_filed": round(float(correct[auto].mean()), 3) if auto.any() else np.nan,
    }


def choose_threshold(confidence, correct) -> float:
    """Lowest threshold whose auto-filed pages reach the target accuracy."""
    for t in THRESHOLDS:
        auto = confidence >= t
        if auto.sum() >= MIN_PAGES_FOR_THRESHOLD and correct[auto].mean() >= TARGET_ACCURACY:
            return float(t)
    return 1.01   # target never reached: send everything to review


# Main

def main() -> None:
    """Run the TF-IDF calibration and routing evaluation."""
    folds = pd.read_csv(FOLDS, dtype={"id": str}).reset_index(drop=True)
    texts = folds["id"].map(lambda i: (TEXTS / f"{i}.txt").read_text(encoding="utf-8", errors="replace")
                            if (TEXTS / f"{i}.txt").exists() else "")
    labels = folds["department"]

    pages, by_fold, raw_list, cal_list = [], [], [], []
    for fold in sorted(folds["fold"].unique()):
        train, test = folds["fold"] != fold, folds["fold"] == fold

        raw = tfidf_model().fit(texts[train], labels[train])
        raw_proba = raw.predict_proba(texts[test])

        cal = calibrated_model().fit(texts[train], labels[train])
        cal_proba = cal.predict_proba(texts[test])
        cal_pred = cal.classes_[cal_proba.argmax(axis=1)]
        cal_conf = cal_proba.max(axis=1)

        # Threshold from the training folds only
        inner = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
        train_proba = cross_val_predict(calibrated_model(), texts[train], labels[train],
                                        cv=inner, method="predict_proba")
        train_classes = np.sort(labels[train].unique())
        train_correct = train_classes[train_proba.argmax(axis=1)] == labels[train].to_numpy()
        threshold = choose_threshold(train_proba.max(axis=1), train_correct)

        true = labels[test].to_numpy()
        correct = cal_pred == true
        routed = np.where(cal_conf >= threshold, "auto-filed", "review")
        by_fold.append({"fold": fold, "threshold_from_training": threshold,
                        **{k: v for k, v in routing_at(cal_conf, correct, threshold).items()
                           if k != "threshold"}})

        pages.append(pd.DataFrame({
            "id": folds.loc[test, "id"], "fold": fold, "true": true,
            "predicted_raw": raw.classes_[raw_proba.argmax(axis=1)],
            "confidence_raw": raw_proba.max(axis=1).round(3),
            "predicted": cal_pred, "confidence": cal_conf.round(3),
            "correct": correct, "routed": routed,
        }))
        raw_list.append(raw_proba)
        cal_list.append(cal_proba)

    # Pool out-of-fold predictions
    raw_probas, cal_probas = np.vstack(raw_list), np.vstack(cal_list)
    pages = pd.concat(pages, ignore_index=True)
    classes = np.sort(labels.unique())
    raw_correct = (pages["predicted_raw"] == pages["true"]).to_numpy()
    cal_correct = pages["correct"].to_numpy()

    calibration = pd.DataFrame([
        {"model": "raw", "accuracy": round(raw_correct.mean(), 3),
         "ece": expected_calibration_error(pages["confidence_raw"].to_numpy(), raw_correct),
         "log_loss": round(log_loss(pages["true"], raw_probas, labels=classes), 3)},
        {"model": "calibrated", "accuracy": round(cal_correct.mean(), 3),
         "ece": expected_calibration_error(pages["confidence"].to_numpy(), cal_correct),
         "log_loss": round(log_loss(pages["true"], cal_probas, labels=classes), 3)},
    ])

    curve = pd.DataFrame(
        [{"model": "raw", **routing_at(pages["confidence_raw"].to_numpy(), raw_correct, t)} for t in THRESHOLDS]
        + [{"model": "calibrated", **routing_at(pages["confidence"].to_numpy(), cal_correct, t)} for t in THRESHOLDS]
    )
    by_fold = pd.DataFrame(by_fold)

    OUT.mkdir(parents=True, exist_ok=True)
    calibration.to_csv(OUT / "calibration_summary.csv", index=False)
    curve.to_csv(OUT / "routing_curve.csv", index=False)
    by_fold.to_csv(OUT / "routing_by_fold.csv", index=False)
    pages.to_csv(OUT / "routing_predictions.csv", index=False)

    auto = pages["routed"] == "auto-filed"
    print("Calibration (lower ECE and log loss = more trustworthy confidence):")
    print(calibration.to_string(index=False))
    print("\nRouting curve, calibrated model (all 250 pages, out-of-fold):")
    print(curve[curve["model"] == "calibrated"].drop(columns="model").to_string(index=False))
    print(f"\nTarget {TARGET_ACCURACY:.0%}, threshold chosen on training folds, applied to each held-out fold:")
    print(by_fold.to_string(index=False))
    print(f"\nOverall: {auto.sum()} of {len(pages)} pages auto-filed ({auto.mean():.0%}), "
          f"accuracy {pages.loc[auto, 'correct'].mean():.1%}; {(~auto).sum()} sent to review")
    print(f"Saved to {OUT}")


if __name__ == "__main__":
    main()
