"""Final test of the frozen pipeline on 150 held-back pages, scored once.

The settings were fixed before any test prediction:
- text: hybrid OCR with the page's own Bates number removed (ZIP+4 codes kept)
- representation: Qwen3-Embedding-0.6B vectors
- classifier: scaled logistic regression, balanced class weights, sigmoid calibration
  (3 inner folds, seed 44), trained on all 250 training pages
- auto-filing cutoff 0.55; a page with fewer than 20 tokens always goes to a person
- record-type suggestion: qwen3:8b with the taxonomy cards, the 5 nearest training pages
  and the classifier's first and second choice
- reference: TF-IDF on the hybrid text, trained on all 250 pages

Stage "predict" never reads the test labels, only the page IDs, and needs Ollama.
Stage "score" opens the labels and refuses to run twice (--force only rebuilds the same
files after a crash). The app's pipeline imports CUTOFF, MIN_TOKENS, K, TRAIN_LABELS and
latest() from here.

Writes results/final_test/: predictions, neighbors, suggestions_A.jsonl and
tfidf_predictions (predict); scores, per_department, rag_comparison, routing_curve,
certainty, excluded_check and scored_pages (score).

Usage:
    python pipeline/05_validate/final_test.py --stage predict
    python pipeline/05_validate/final_test.py --stage score
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.stats import binomtest
from sklearn.metrics import accuracy_score, f1_score

import sys
sys.path[:0] = [str(Path(__file__).resolve().parents[2] / "pipeline" / step) for step in ("03_classify", "04_route",)]  # sibling pipeline steps
from rag_suggest import (MERGE_FOR_SCORING, OUT_OF_SCOPE, TAXONOMY, ask, build_cards,
                         build_prompt, department_of)
from review_routing import THRESHOLDS, routing_at
from train_embeddings import EMB_DIR, MODEL_TAG, calibrated_model, load_vectors
from train_tfidf import TEXT_DIRS, read_texts, tfidf_model

ROOT = Path(__file__).resolve().parents[2]
FOLDS = ROOT / "results" / "cv_folds.csv"
TEST_LABELS = ROOT / "labels" / "final_test_labels_salo.csv"
TRAIN_LABELS = ROOT / "labels" / "label_pm_labels_salo.csv"
OUT = ROOT / "results" / "final_test"

# Frozen settings
CUTOFF = 0.55
MIN_TOKENS = 20
K = 5
SEED = 44
BOOTSTRAP = 2000


def latest(path: Path) -> pd.DataFrame:
    """Read a labels file, keeping each document's latest answer."""
    df = pd.read_csv(path, dtype=str)
    return df.sort_values("saved_at").drop_duplicates("doc_id", keep="last")


# Stage 1: predict (no test labels read)

def predict() -> None:
    """Stage 1: predict the test pages without reading their labels."""
    OUT.mkdir(parents=True, exist_ok=True)
    train = pd.read_csv(FOLDS, dtype={"id": str}).reset_index(drop=True)
    test_ids = latest(TEST_LABELS)["doc_id"].reset_index(drop=True)   # IDs only
    assert not set(test_ids) & set(train["id"]), "test page found in training"

    X_train = load_vectors(train["id"], "hybrid")
    y_train = train["department"].to_numpy()
    X_test = load_vectors(test_ids, "hybrid")
    tokens = (pd.read_csv(EMB_DIR / f"hybrid_{MODEL_TAG}_ids.csv", dtype={"id": str})
              .set_index("id")["tokens"])

    # Frozen classifier, trained once on all 250
    model = calibrated_model().fit(X_train, y_train)
    proba = model.predict_proba(X_test)
    order = np.argsort(-proba, axis=1)
    first, second = model.classes_[order[:, 0]], model.classes_[order[:, 1]]
    conf = proba[np.arange(len(test_ids)), order[:, 0]]
    conf2 = proba[np.arange(len(test_ids)), order[:, 1]]
    page_tokens = test_ids.map(tokens).fillna(0).astype(int).to_numpy()
    thin = page_tokens < MIN_TOKENS
    routed = np.where(thin, "review", np.where(conf >= CUTOFF, "auto-filed", "review"))
    reason = np.where(thin, "under 20 tokens", np.where(conf >= CUTOFF, "confident", "below cutoff"))

    preds = pd.DataFrame({"id": test_ids, "tokens": page_tokens,
                          "predicted": first, "confidence": conf.round(3),
                          "second": second, "second_confidence": conf2.round(3),
                          "routed": routed, "route_reason": reason})
    preds.to_csv(OUT / "predictions.csv", index=False)
    print(f"Classifier: {len(preds)} test pages | auto-filed {(routed == 'auto-filed').sum()} | "
          f"review {(routed == 'review').sum()} (of which {thin.sum()} under {MIN_TOKENS} tokens)")

    # Neighbors from all 250 training pages (vectors are normalized: dot = cosine)
    sims = X_test @ X_train.T
    nb_rows = []
    for row, doc_id in enumerate(test_ids):
        for rank, j in enumerate(np.argsort(-sims[row])[:K], 1):
            nb_rows.append({"id": doc_id, "rank": rank, "neighbor_id": train["id"][j],
                            "neighbor_department": y_train[j], "similarity": round(float(sims[row, j]), 4)})
    neighbors = pd.DataFrame(nb_rows)
    neighbors.to_csv(OUT / "neighbors.csv", index=False)

    # TF-IDF reference, trained once on all 250
    tf = tfidf_model().fit(read_texts(TEXT_DIRS["hybrid"], train["id"]), y_train)
    pd.DataFrame({"id": test_ids, "predicted": tf.predict(read_texts(TEXT_DIRS["hybrid"], test_ids))}
                 ).to_csv(OUT / "tfidf_predictions.csv", index=False)

    # RAG variant A on the review queue (resumable)
    cards, types = build_cards(yaml.safe_load(TAXONOMY.read_text(encoding="utf-8")))
    system = ("You are a records clerk at Philip Morris USA filing scanned documents into the company's "
              "records taxonomy. Use only the evidence given. Answer in JSON.\n\n" + cards)
    train_labels = latest(TRAIN_LABELS)
    label_of = dict(zip(train_labels["doc_id"], train_labels["type_id"]))
    choices = {r.id: [(r.predicted, r.confidence), (r.second, r.second_confidence)]
               for r in preds.itertuples()}

    out_path = OUT / "suggestions_A.jsonl"
    done = set()
    if out_path.exists():
        done = {json.loads(l)["id"] for l in out_path.read_text(encoding="utf-8").splitlines() if l.strip()}
    queue = preds.loc[preds["routed"] == "review", "id"].tolist()
    todo = [i for i in queue if i not in done]
    print(f"RAG A: {len(queue)} review pages, {len(done)} done, {len(todo)} to do", flush=True)

    type_ids = sorted(types)
    start = time.time()
    with out_path.open("a", encoding="utf-8") as f:
        for n, doc_id in enumerate(todo, 1):
            nb = neighbors[neighbors["id"] == doc_id].sort_values("rank")
            t0 = time.time()
            try:
                ans = ask(system, build_prompt(doc_id, nb, label_of, types, choices, "A"), type_ids)
            except Exception as e:
                print(f"{doc_id}: failed ({str(e)[:120]})", flush=True)
                continue
            rec = {"id": doc_id, "department": department_of(ans["record_type"]),
                   "second_department": department_of(ans["second_record_type"]),
                   "seconds": round(time.time() - t0, 2), **ans}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            if n % 10 == 0 or n == len(todo):
                per = (time.time() - start) / n
                print(f"{n}/{len(todo)} | {per:.1f} s per page | about {per * (len(todo) - n) / 60:.0f} min left",
                      flush=True)

    answered = sum(1 for l in out_path.read_text(encoding="utf-8").splitlines() if l.strip())
    print(f"\nRAG answers saved: {answered} of {len(queue)}. "
          f"{'Ready for --stage score.' if answered == len(queue) else 'Rerun --stage predict to finish.'}")


# Stage 2: score once

def interval(values) -> tuple[float, float]:
    """Return the 95% percentile interval of the values, ignoring NaN."""
    v = np.asarray([x for x in values if not np.isnan(x)])
    return (round(float(np.percentile(v, 2.5)), 3), round(float(np.percentile(v, 97.5)), 3)) if len(v) else (np.nan, np.nan)


def score(force: bool) -> None:
    """Stage 2: score the saved predictions once."""
    if (OUT / "scores.csv").exists() and not force:
        raise SystemExit("Scores already exist: the final test is scored once (D-62). "
                         "Use --force only to rebuild the same files after a crash.")

    preds = pd.read_csv(OUT / "predictions.csv", dtype={"id": str})
    rag = pd.DataFrame([json.loads(l) for l in (OUT / "suggestions_A.jsonl").read_text(encoding="utf-8").splitlines()
                        if l.strip()])
    queue = preds.loc[preds["routed"] == "review", "id"]
    missing = set(queue) - set(rag["id"])
    if missing:
        raise SystemExit(f"{len(missing)} review pages have no RAG answer yet: rerun --stage predict first.")

    # Labels opened here for the first time
    labels = latest(TEST_LABELS)
    labels["true"] = labels["function_id"].replace(MERGE_FOR_SCORING)
    df = (preds.merge(labels[["doc_id", "outcome", "true", "certainty"]], left_on="id", right_on="doc_id")
          .merge(pd.read_csv(OUT / "tfidf_predictions.csv", dtype={"id": str})
                 .rename(columns={"predicted": "tfidf"}), on="id")
          .merge(rag[["id", "department", "second_department"]]
                 .rename(columns={"department": "rag", "second_department": "rag_second"}), on="id", how="left"))

    excluded = df[df["outcome"] != "labeled"]
    v = df[(df["outcome"] == "labeled") & ~df["true"].isin(OUT_OF_SCOPE)].reset_index(drop=True)
    v["correct"] = v["predicted"] == v["true"]
    auto = (v["routed"] == "auto-filed").to_numpy()
    review = ~auto
    v["pipeline"] = np.where(auto, v["predicted"], v["rag"])   # if every suggestion were accepted
    y, p = v["true"].to_numpy(), v["predicted"].to_numpy()

    def stats(idx):
        yy, pp, aa, cc = y[idx], p[idx], auto[idx], v["correct"].to_numpy()[idx]
        return {"macro_f1": f1_score(yy, pp, average="macro", zero_division=0),
                "accuracy": accuracy_score(yy, pp),
                "auto_share": aa.mean(),
                "auto_accuracy": cc[aa].mean() if aa.any() else np.nan,
                "pipeline_accuracy": (v["pipeline"].to_numpy()[idx] == yy).mean(),
                "tfidf_macro_f1": f1_score(yy, v["tfidf"].to_numpy()[idx], average="macro", zero_division=0)}

    point = stats(np.arange(len(v)))
    rng = np.random.default_rng(SEED)
    boots = [stats(rng.integers(0, len(v), len(v))) for _ in range(BOOTSTRAP)]
    rows = []
    for name, value in point.items():
        lo, hi = interval([b[name] for b in boots])
        rows.append({"measure": name, "value": round(float(value), 3), "ci_low": lo, "ci_high": hi})
    rows.append({"measure": "pages_scored", "value": len(v), "ci_low": np.nan, "ci_high": np.nan})
    rows.append({"measure": "auto_filed_pages", "value": int(auto.sum()), "ci_low": np.nan, "ci_high": np.nan})
    scores = pd.DataFrame(rows)

    # Per-department F1 with intervals
    depts = sorted(set(y))
    dep_point = f1_score(y, p, labels=depts, average=None, zero_division=0)
    dep_boot = []
    rng = np.random.default_rng(SEED)
    for _ in range(BOOTSTRAP):
        i = rng.integers(0, len(v), len(v))
        dep_boot.append(f1_score(y[i], p[i], labels=depts, average=None, zero_division=0))
    dep_boot = np.array(dep_boot)
    per_dept = pd.DataFrame({"department": depts, "pages": [int((y == d).sum()) for d in depts],
                             "f1": dep_point.round(3),
                             "ci_low": np.percentile(dep_boot, 2.5, axis=0).round(3),
                             "ci_high": np.percentile(dep_boot, 97.5, axis=0).round(3)})

    # RAG A vs classifier on the test review queue
    q = v[review]
    cls1 = q["predicted"] == q["true"]
    cls2 = cls1 | (q["second"] == q["true"])
    rag1 = q["rag"] == q["true"]
    rag2 = rag1 | (q["rag_second"] == q["true"])
    b, c = int((rag1 & ~cls1).sum()), int((~rag1 & cls1).sum())
    p_value = binomtest(b, b + c, 0.5).pvalue if b + c else np.nan
    rag_cmp = pd.DataFrame([{"review_pages": len(q), "classifier_top1": round(cls1.mean(), 3),
                             "rag_top1": round(rag1.mean(), 3), "classifier_top2": round(cls2.mean(), 3),
                             "rag_top2": round(rag2.mean(), 3), "rag_only_right": b, "classifier_only_right": c,
                             "mcnemar_exact_p": round(float(p_value), 4)}])

    # Accuracy by the labeler's certainty; test routing curve (description only)
    certainty = (v.groupby("certainty")["correct"].agg(pages="size", accuracy="mean").round(3).reset_index())
    curve = pd.DataFrame([routing_at(v["confidence"].to_numpy(), v["correct"].to_numpy(), t) for t in THRESHOLDS])
    excl = excluded[["id", "tokens", "predicted", "confidence", "routed", "route_reason"]]

    OUT.mkdir(parents=True, exist_ok=True)
    scores.to_csv(OUT / "scores.csv", index=False)
    per_dept.to_csv(OUT / "per_department.csv", index=False)
    rag_cmp.to_csv(OUT / "rag_comparison.csv", index=False)
    certainty.to_csv(OUT / "certainty.csv", index=False)
    curve.to_csv(OUT / "routing_curve.csv", index=False)
    excl.to_csv(OUT / "excluded_check.csv", index=False)
    v.drop(columns=["doc_id"]).to_csv(OUT / "scored_pages.csv", index=False)

    print(f"FINAL TEST (scored once, D-62): {len(v)} valid pages, 95% bootstrap intervals\n")
    print(scores.to_string(index=False))
    print("\nPer department:\n" + per_dept.to_string(index=False))
    print("\nRAG A vs classifier on the review queue:\n" + rag_cmp.to_string(index=False))
    print("\nBy labeler certainty:\n" + certainty.to_string(index=False))
    print("\nRouting curve on the test set (description only, cutoff stays 0.55):\n" + curve.to_string(index=False))
    print("\nExcluded pages (should go to a person):\n" + excl.to_string(index=False))
    print(f"\nSaved to {OUT}")


def main() -> None:
    """Run one stage."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["predict", "score"], required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    predict() if args.stage == "predict" else score(args.force)


if __name__ == "__main__":
    main()
