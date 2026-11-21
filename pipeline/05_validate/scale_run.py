"""Scale run: the frozen pipeline on the 1,000-document scale batch, without labels.

How many pages does LocalRIM file on its own, how many go to a person and why, and how
long does it take on one PC? Same frozen settings as final_test.py, imported from it:
hybrid text, Qwen3 embeddings, calibrated classifier trained on all 250 pages, cutoff
0.55, pages under 20 tokens to a person, and a record-type suggestion for every page
sent to a person. Nothing is scored against labels here.

Also:
- pages with no readable text go to a person as "no readable text" and are counted;
- expected errors among auto-filed pages are projected from the final test's measured
  accuracy and labelled as a projection;
- a drift check of confidences and the auto-filed share against the final test;
- a random sample of 20 auto-filed pages for the blind spot-check (spotcheck.py); once
  labels/spotcheck_<annotator>.csv exists, the agreement is reported.

Reads the manifest (positions 404-1403), data/pm_sample/ocr_text/sources.csv,
results/ocr_eda/C and D pages.csv (OCR seconds) and what final_test.py uses.
Writes results/scale_run/: predictions, neighbors, suggestions_A.jsonl, summary,
departments, timing, spotcheck_sample and spotcheck_result. Resumable.

Usage:
    python pipeline/05_validate/scale_run.py --limit 5
    python pipeline/05_validate/scale_run.py
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from final_test import CUTOFF, K, MIN_TOKENS, SEED, TRAIN_LABELS, latest
import sys
sys.path[:0] = [str(Path(__file__).resolve().parents[2] / "pipeline" / step) for step in ("03_classify", "04_route",)]  # sibling pipeline steps
from rag_suggest import MERGE_FOR_SCORING, TAXONOMY, ask, build_cards, build_prompt, department_of
from train_embeddings import EMB_DIR, MODEL_TAG, calibrated_model, load_vectors

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "pm_sample"
MANIFEST = DATA / "manifest.csv"
SOURCES = DATA / "ocr_text" / "sources.csv"
TEXTS = DATA / "ocr_text_clean" / "hybrid"
FOLDS = ROOT / "results" / "cv_folds.csv"
FINAL = ROOT / "results" / "final_test"
SPOT_LABELS = ROOT / "labels" / "spotcheck_salo.csv"
OUT = ROOT / "results" / "scale_run"

SCALE_FIRST, SCALE_LAST = 404, 1403          # manifest positions of the scale batch
TEST_AUTO_ACCURACY = (0.871, 0.781, 0.944)   # final test: value, interval low, high
EMBED_SECONDS_PER_PAGE = 55 / 1400           # embed_pages.py (1,400 pages in 55 s)
SPOTCHECK_PAGES = 20


def has_text(doc_id: str) -> bool:
    """Return whether a page has readable text."""
    path = TEXTS / f"{doc_id}.txt"
    return path.exists() and bool(path.read_text(encoding="utf-8", errors="replace").strip())


def main() -> None:
    """Run the frozen pipeline on the scale batch and write the results."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="RAG on only this many review pages (a trial)")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    # Pages
    manifest = pd.read_csv(MANIFEST, dtype=str)
    manifest["position"] = np.arange(1, len(manifest) + 1)
    batch = manifest[manifest["position"].between(SCALE_FIRST, SCALE_LAST)]
    emb_ids = pd.read_csv(EMB_DIR / f"hybrid_{MODEL_TAG}_ids.csv", dtype={"id": str}).set_index("id")
    readable = batch["id"].isin(emb_ids.index)
    ids = batch.loc[readable, "id"].reset_index(drop=True)
    test_ids = set(pd.read_csv(FINAL / "predictions.csv", dtype={"id": str})["id"])
    print(f"Scale batch: {len(batch)} documents, {len(ids)} readable pages "
          f"({(~readable).sum()} without a page image), {len(set(ids) & test_ids)} of them final-test pages")

    # Classifier and routing (frozen)
    train = pd.read_csv(FOLDS, dtype={"id": str}).reset_index(drop=True)
    X_train, y_train = load_vectors(train["id"], "hybrid"), train["department"].to_numpy()
    t0 = time.time()
    model = calibrated_model().fit(X_train, y_train)
    fit_seconds = time.time() - t0

    t0 = time.time()
    X = load_vectors(ids, "hybrid")
    proba = model.predict_proba(X)
    order = np.argsort(-proba, axis=1)
    rows = np.arange(len(ids))
    first, second = model.classes_[order[:, 0]], model.classes_[order[:, 1]]
    conf, conf2 = proba[rows, order[:, 0]], proba[rows, order[:, 1]]
    tokens = ids.map(emb_ids["tokens"]).astype(int).to_numpy()
    text_ok = ids.map(has_text).to_numpy()
    reason = np.select([~text_ok, tokens < MIN_TOKENS, conf >= CUTOFF],
                       ["no readable text", "under 20 tokens", "confident"], "below cutoff")
    routed = np.where(reason == "confident", "auto-filed", "review")
    classify_seconds = (time.time() - t0) / len(ids)

    preds = pd.DataFrame({"id": ids, "tokens": tokens, "predicted": first, "confidence": conf.round(3),
                          "second": second, "second_confidence": conf2.round(3),
                          "routed": routed, "route_reason": reason, "final_test_page": ids.isin(test_ids)})
    preds.to_csv(OUT / "predictions.csv", index=False)

    # Consistency: test pages must match the final test exactly (same frozen model)
    ft = pd.read_csv(FINAL / "predictions.csv", dtype={"id": str}).set_index("id")
    both = preds[preds["final_test_page"]].set_index("id")
    same = ((both["predicted"] == ft.loc[both.index, "predicted"]) &
            (both["routed"] == ft.loc[both.index, "routed"])).mean()
    print(f"Consistency with the final test: {same:.0%} of {len(both)} test pages identical")

    # Neighbours
    sims = X @ X_train.T
    nb = []
    for r, doc_id in enumerate(ids):
        for rank, j in enumerate(np.argsort(-sims[r])[:K], 1):
            nb.append({"id": doc_id, "rank": rank, "neighbor_id": train["id"][j],
                       "neighbor_department": y_train[j], "similarity": round(float(sims[r, j]), 4)})
    neighbors = pd.DataFrame(nb)
    neighbors.to_csv(OUT / "neighbors.csv", index=False)

    # Record-type suggestions on the review queue (resumable)
    cards, types = build_cards(yaml.safe_load(TAXONOMY.read_text(encoding="utf-8")))
    system = ("You are a records clerk at Philip Morris USA filing scanned documents into the company's "
              "records taxonomy. Use only the evidence given. Answer in JSON.\n\n" + cards)
    labels = latest(TRAIN_LABELS)
    label_of = dict(zip(labels["doc_id"], labels["type_id"]))
    choices = {r.id: [(r.predicted, r.confidence), (r.second, r.second_confidence)] for r in preds.itertuples()}

    out_path = OUT / "suggestions_A.jsonl"
    done = set()
    if out_path.exists():
        done = {json.loads(l)["id"] for l in out_path.read_text(encoding="utf-8").splitlines() if l.strip()}
    queue = preds.loc[preds["routed"] == "review", "id"].tolist()
    todo = [i for i in queue if i not in done]
    if args.limit:
        todo = todo[: args.limit]
    print(f"Routing: {len(queue)} of {len(preds)} pages to a person. RAG A: {len(done)} done, {len(todo)} to do "
          f"(about {len(todo) * 5.3 / 60:.0f} min)", flush=True)

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
                                "seconds": round(time.time() - t0, 2), **ans}, ensure_ascii=False) + "\n")
            f.flush()
            if n % 25 == 0 or n == len(todo):
                per = (time.time() - start) / n
                print(f"{n}/{len(todo)} | {per:.1f} s per page | about {per * (len(todo) - n) / 60:.0f} min left",
                      flush=True)

    rag = pd.DataFrame([json.loads(l) for l in out_path.read_text(encoding="utf-8").splitlines() if l.strip()])
    summarize(preds, rag, len(queue), fit_seconds, classify_seconds)


def summarize(preds, rag, queue_size, fit_seconds, classify_seconds) -> None:
    """Build and save the run's summary tables."""
    n = len(preds)
    auto = preds["routed"] == "auto-filed"
    reasons = preds["route_reason"].value_counts()
    acc, lo, hi = TEST_AUTO_ACCURACY
    ft = pd.read_csv(FINAL / "predictions.csv", dtype={"id": str})
    q = lambda s: s.quantile([0.25, 0.5, 0.75]).round(3).tolist()

    merged = preds.merge(rag[["id", "department"]], on="id", how="left") if len(rag) else preds.assign(department=np.nan)
    answered = merged["department"].notna()
    rag_agrees = (merged.loc[answered, "department"] == merged.loc[answered, "predicted"]).mean() if answered.any() else np.nan

    summary = pd.DataFrame([
        ("pages", n), ("auto_filed", int(auto.sum())), ("auto_filed_share", round(auto.mean(), 3)),
        ("to_a_person", int((~auto).sum())),
        ("reason_below_cutoff", int(reasons.get("below cutoff", 0))),
        ("reason_under_20_tokens", int(reasons.get("under 20 tokens", 0))),
        ("reason_no_readable_text", int(reasons.get("no readable text", 0))),
        ("projected_wrong_auto_filed", round(auto.sum() * (1 - acc), 1)),
        ("projected_wrong_low", round(auto.sum() * (1 - hi), 1)),
        ("projected_wrong_high", round(auto.sum() * (1 - lo), 1)),
        ("final_test_auto_filed_share", round((ft["routed"] == "auto-filed").mean(), 3)),
        ("confidence_quartiles_scale", q(preds["confidence"])),
        ("confidence_quartiles_final_test", q(ft["confidence"])),
        ("rag_answered", f"{int(answered.sum())} of {queue_size}"),
        ("rag_agrees_with_classifier_first", round(float(rag_agrees), 3) if answered.any() else np.nan),
    ], columns=["measure", "value"])

    # Department mix: predicted (all pages, auto-filed) vs training labels
    train = pd.read_csv(FOLDS)["department"].value_counts(normalize=True)
    deps = pd.DataFrame({
        "training_share": train,
        "predicted_share_all": preds["predicted"].value_counts(normalize=True),
        "auto_filed_pages": preds.loc[auto, "predicted"].value_counts(),
        "rag_suggested_share": merged.loc[answered, "department"].value_counts(normalize=True),
    }).fillna(0).round(3).rename_axis("department").reset_index()

    # Time per stage (seconds per page); OCR from the full OCR runs, restricted to this batch.
    ids = set(preds["id"])
    src = pd.read_csv(SOURCES, dtype=str)
    c_sec = pd.read_csv(ROOT / "results/ocr_eda/C/pages.csv", dtype={"id": str}, usecols=["id", "seconds"])
    d_sec = pd.read_csv(ROOT / "results/ocr_eda/D/pages.csv", dtype={"id": str}, usecols=["id", "seconds"])
    d_pages = set(src.loc[src["hybrid_source"] == "D", "id"]) & ids
    ocr_c = c_sec.loc[c_sec["id"].isin(ids), "seconds"].mean()
    ocr_d_total = d_sec.loc[d_sec["id"].isin(d_pages), "seconds"].sum()
    rag_mean = rag["seconds"].mean() if len(rag) else np.nan
    stages = [
        ("OCR, PaddleOCR (every page)", ocr_c, "measured Nov 8"),
        ("OCR, vision model (hard pages only, spread over all)", ocr_d_total / n, f"{len(d_pages)} pages, measured Nov 9"),
        ("Embedding", EMBED_SECONDS_PER_PAGE, "measured Nov 12"),
        ("Classifier and routing", classify_seconds, "measured now (model fitted once in %.0f s)" % fit_seconds),
        ("RAG suggestion (pages to a person only, spread over all)", rag_mean * (~auto).sum() / n if len(rag) else np.nan,
         f"{rag_mean:.1f} s per review page, measured now" if len(rag) else "not run yet"),
    ]
    timing = pd.DataFrame(stages, columns=["stage", "seconds_per_page", "source"])
    timing["seconds_per_page"] = timing["seconds_per_page"].round(3)
    total = timing["seconds_per_page"].sum()
    timing.loc[len(timing)] = ["Total", round(total, 2), f"about {total * 1000 / 3600:.1f} hours per 1,000 pages"]

    # Optional QA: blind spot-check sample (auto-filed, not final-test pages), drawn once
    sample_path = OUT / "spotcheck_sample.csv"
    if not sample_path.exists():
        pool = preds[auto & ~preds["final_test_page"]]
        pool.sample(min(SPOTCHECK_PAGES, len(pool)), random_state=SEED)[["id"]].to_csv(sample_path, index=False)
    spot = None
    if SPOT_LABELS.exists():
        s = latest(SPOT_LABELS)
        s = s[s["outcome"] == "labeled"].copy()
        s["true"] = s["function_id"].replace(MERGE_FOR_SCORING)
        s = s.merge(preds[["id", "predicted"]], left_on="doc_id", right_on="id")
        s["agrees"] = s["true"] == s["predicted"]
        s[["doc_id", "true", "predicted", "agrees", "certainty"]].to_csv(OUT / "spotcheck_result.csv", index=False)
        spot = f"{int(s['agrees'].sum())} of {len(s)} spot-checked auto-filed pages match your blind label"

    summary.to_csv(OUT / "summary.csv", index=False)
    deps.to_csv(OUT / "departments.csv", index=False)
    timing.to_csv(OUT / "timing.csv", index=False)

    print("\nSCALE RUN (frozen pipeline, no labels)\n")
    print(summary.to_string(index=False))
    print("\nDepartments (shares):\n" + deps.to_string(index=False))
    print("\nTime per page:\n" + timing.to_string(index=False))
    print(f"\nProjected misfiled among auto-filed pages: about {summary.loc[7, 'value']} "
          f"(range {summary.loc[8, 'value']}-{summary.loc[9, 'value']}), from the final test's accuracy; a projection, not a measurement.")
    if spot:
        print("Spot-check (descriptive): " + spot)
    print(f"\nSaved to {OUT}")


if __name__ == "__main__":
    main()
