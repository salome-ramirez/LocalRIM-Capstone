"""Record-type suggestions for the pages that filed themselves.

The RAG runs asked the model only about pages sent to review. The app's Archive needs a
record type for every filed page, so this asks the same model, with the same prompt,
cards, settings and similar pages, about the auto-filed pages too:
    training  the 56 auto-filed training pages, cross-validated like rag_suggest.py
              -> results/rag/suggestions_A_autofiled.jsonl
    scale     the 428 auto-filed scale-run pages (including the 62 auto-filed final-test
              pages), with the final model and index as in scale_run.py
              -> results/scale_run/suggestions_A_autofiled.jsonl
Resumable. Scoring is done by score_record_types.py.

Usage:
    python pipeline/04_route/rag_autofiled.py --limit 3
    python pipeline/04_route/rag_autofiled.py
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pandas as pd
import yaml

import sys
sys.path[:0] = [str(Path(__file__).resolve().parents[2] / "pipeline" / step) for step in ("05_validate",)]  # sibling pipeline steps
from final_test import TRAIN_LABELS, latest
from rag_suggest import TAXONOMY, ask, build_cards, build_prompt, classifier_choices, department_of

ROOT = Path(__file__).resolve().parents[2]
FOLDS = ROOT / "results" / "cv_folds.csv"
CV_PREDS = ROOT / "results" / "classifier" / "emb_cv_predictions.csv"
RAG = ROOT / "results" / "rag"
SCALE = ROOT / "results" / "scale_run"


def run(name, ids, neighbors, choices, label_of, types, system, out_path):
    """Ask the model about each page not yet answered and append the answers."""
    done = set()
    if out_path.exists():
        done = {json.loads(l)["id"] for l in out_path.read_text(encoding="utf-8").splitlines() if l.strip()}
    todo = [i for i in ids if i not in done]
    print(f"{name}: {len(ids)} auto-filed pages, {len(done)} done, {len(todo)} to do", flush=True)
    type_ids = sorted(types)
    start = time.time()
    with out_path.open("a", encoding="utf-8") as f:
        for n, doc_id in enumerate(todo, 1):
            t0 = time.time()
            try:
                ans = ask(system, build_prompt(doc_id, neighbors[neighbors["id"] == doc_id].sort_values("rank"),
                                               label_of, types, choices, "A"), type_ids)
            except Exception as e:
                print(f"{doc_id}: failed ({str(e)[:120]})", flush=True)
                continue
            f.write(json.dumps({"id": doc_id, "department": department_of(ans["record_type"]),
                                "second_department": department_of(ans["second_record_type"]),
                                "classifier_first": choices[doc_id][0][0],
                                "seconds": round(time.time() - t0, 2), **ans}, ensure_ascii=False) + "\n")
            f.flush()
            if n % 25 == 0 or n == len(todo):
                per = (time.time() - start) / n
                print(f"{name} {n}/{len(todo)} | {per:.1f} s per page | about {per * (len(todo) - n) / 60:.0f} min left",
                      flush=True)


def main() -> None:
    """Run both groups."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="only this many pages per group (a trial)")
    args = parser.parse_args()

    cards, types = build_cards(yaml.safe_load(TAXONOMY.read_text(encoding="utf-8")))
    system = ("You are a records clerk at Philip Morris USA filing scanned documents into the company's "
              "records taxonomy. Use only the evidence given. Answer in JSON.\n\n" + cards)
    labels = latest(TRAIN_LABELS)
    label_of = dict(zip(labels["doc_id"], labels["type_id"]))

    # Training pages (cross-validated, as rag_suggest.py)
    folds = pd.read_csv(FOLDS, dtype={"id": str})
    cv = pd.read_csv(CV_PREDS, dtype={"id": str})
    train_ids = cv.loc[cv["routed"] == "auto-filed", "id"].tolist()[: args.limit]
    print("Fitting the fold models for the classifier's choices (same as rag_suggest.py)...", flush=True)
    train_choices = classifier_choices(folds)
    train_nb = pd.read_csv(RAG / "neighbors.csv", dtype={"id": str, "neighbor_id": str})
    run("training", train_ids, train_nb, train_choices, label_of, types, system,
        RAG / "suggestions_A_autofiled.jsonl")

    # Scale-run pages (final model and index, as scale_run.py)
    preds = pd.read_csv(SCALE / "predictions.csv", dtype={"id": str})
    scale_ids = preds.loc[preds["routed"] == "auto-filed", "id"].tolist()[: args.limit]
    scale_choices = {r.id: [(r.predicted, r.confidence), (r.second, r.second_confidence)]
                     for r in preds.itertuples()}
    scale_nb = pd.read_csv(SCALE / "neighbors.csv", dtype={"id": str, "neighbor_id": str})
    run("scale", scale_ids, scale_nb, scale_choices, label_of, types, system,
        SCALE / "suggestions_A_autofiled.jsonl")
    print("Done. Score against the labels in a separate step.")


if __name__ == "__main__":
    main()
