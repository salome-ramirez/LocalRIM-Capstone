"""Record-type suggestions from a local language model for the pages a person reviews.

The model (qwen3:8b through Ollama; temperature 0, fixed seed, thinking off, JSON output
limited to the taxonomy's record-type IDs) gets the page's text, the five most similar
filed pages from the other folds with their record types, a card for every department in
scope with the rules R1-R10 and, in variant A, the classifier's first and second choice.
It returns a record type, a second choice from another department, a one-line reason and
the sources it used. Variant C leaves out the classifier's choices.

Reads results/cv_folds.csv, results/rag/neighbors.csv,
results/classifier/emb_cv_predictions.csv, the training labels, the cleaned hybrid text
and the taxonomy.
Writes results/rag/taxonomy_cards.txt, suggestions_<variant>.jsonl (resumable) and
rag_summary_<variant>.csv.

Usage:
    python -m pipeline.04_route.rag_suggest --variant A --limit 3
    python -m pipeline.04_route.rag_suggest --variant A
    python -m pipeline.04_route.rag_suggest --variant C
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np
import pandas as pd
import requests
import yaml

from pipeline.common import MERGE_FOR_SCORING, Progress, ROOT, TRAIN_LABELS as LABELS, done_ids
from train_embeddings import calibrated_model, load_vectors

FOLDS = ROOT / "results" / "cv_folds.csv"
RAG = ROOT / "results" / "rag"
ROUTED = ROOT / "results" / "classifier" / "emb_cv_predictions.csv"
TEXTS = ROOT / "data" / "pm_sample" / "ocr_text_clean" / "hybrid"
TAXONOMY = ROOT / "taxonomy" / "rim_taxonomy_final.yaml"

OLLAMA = "http://localhost:11434/api/chat"
MODEL = "qwen3:8b"
SEED = 44
PAGE_CHARS = 2500
NEIGHBOR_CHARS = 350
NUM_CTX = 8192
OUT_OF_SCOPE = {"HR"}                # HR is out of scope


# Taxonomy cards

def build_cards(tax: dict) -> tuple[str, dict]:
    """One card per department (HR left out) plus the rules; and a type-id lookup."""
    lines, types = [], {}
    for f in tax["functions"]:
        if f["id"] in OUT_OF_SCOPE:
            continue
        lines.append(f"## {f['name']} ({f['id']})")
        lines.append(f"Question: {f['question']}")
        lines.append(f"Definition: {f['definition']}")
        for h in f.get("hints", []):
            lines.append(f"Hint: {h}")
        for s in f["series"]:
            for t in s["types"]:
                types[t["id"]] = f"{f['name']} > {s['name']} > {t['name']}"
                lines.append(f"- {t['id']} {s['name']} > {t['name']}: {t['looks_like']}")
        lines.append("")
    lines.append("## Rules")
    for r in tax["labeling_rules"]:
        lines.append(f"{r['id']}: {r['rule']}")
    return "\n".join(lines), types


def department_of(type_id: str) -> str:
    """Return the scoring department of a record type ID."""
    dept = type_id.split(".")[0]
    return MERGE_FOR_SCORING.get(dept, dept)


# Evidence

def read_text(doc_id: str, limit: int) -> str:
    """Return the first limit characters of a page's cleaned text, whitespace collapsed."""
    path = TEXTS / f"{doc_id}.txt"
    text = path.read_text(encoding="utf-8", errors="replace") if path.exists() else ""
    text = " ".join(text.split())
    return text[:limit] if text else "(no readable text)"


def classifier_choices(folds: pd.DataFrame) -> dict:
    """First and second choice with confidence, same calibrated model as the routing."""
    X = load_vectors(folds["id"], "hybrid")
    y = folds["department"].to_numpy()
    out = {}
    for fold in sorted(folds["fold"].unique()):
        train = (folds["fold"] != fold).to_numpy()
        model = calibrated_model().fit(X[train], y[train])
        proba = model.predict_proba(X[~train])
        for doc_id, p in zip(folds.loc[~train, "id"], proba):
            order = np.argsort(-p)
            out[doc_id] = [(model.classes_[order[k]], float(p[order[k]])) for k in (0, 1)]
    return out


def build_prompt(doc_id, neighbors, label_of, types, choices, variant):
    """Return the prompt for one page: its text, similar filed pages and, for variant A, the classifier's choices."""
    parts = ["PAGE TO FILE (first page of a scanned Philip Morris document; OCR text):",
             read_text(doc_id, PAGE_CHARS), "", "SIMILAR PAGES ALREADY FILED (most similar first):"]
    for k, row in enumerate(neighbors.itertuples(), 1):
        t = label_of.get(row.neighbor_id, "")
        parts.append(f"S{k}. Filed as {t} {types.get(t, '')}. Similarity {row.similarity:.2f}. "
                     f"Text: {read_text(row.neighbor_id, NEIGHBOR_CHARS)}")
    if variant == "A":
        (d1, p1), (d2, p2) = choices[doc_id]
        parts += ["", f"CLASSIFIER (department level; Administration counted with Legal): "
                      f"first choice {d1} ({p1:.0%} confident), second choice {d2} ({p2:.0%})."]
    parts += ["", "Which Philip Morris department would keep the official copy of this page, and which "
                  "record type is it? Decide by the page's purpose, using the cards and rules. Give your "
                  "best record type, a second choice from a DIFFERENT department, a one-line reason, and "
                  "the sources you relied on (S1-S5, rule IDs such as R9"
                  + (", CLASSIFIER" if variant == "A" else "") + ")."]
    return "\n".join(parts)


# Model call

def ask(system: str, prompt: str, type_ids: list[str]) -> dict:
    """Ask the model for a record type; the answer is forced to JSON with valid type IDs."""
    schema = {"type": "object", "properties": {
        "record_type": {"type": "string", "enum": type_ids},
        "second_record_type": {"type": "string", "enum": type_ids},
        "reason": {"type": "string"},
        "sources": {"type": "array", "items": {"type": "string"}}},
        "required": ["record_type", "second_record_type", "reason", "sources"]}
    r = requests.post(OLLAMA, timeout=300, json={
        "model": MODEL, "stream": False, "think": False, "format": schema,
        "options": {"temperature": 0, "seed": SEED, "num_ctx": NUM_CTX},
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": prompt}]})
    r.raise_for_status()
    return json.loads(r.json()["message"]["content"])


# Main

def main() -> None:
    """Run one variant on the cross-validation review queue and summarise it."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=["A", "C"], required=True)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    cards, types = build_cards(yaml.safe_load(TAXONOMY.read_text(encoding="utf-8")))
    RAG.mkdir(parents=True, exist_ok=True)
    (RAG / "taxonomy_cards.txt").write_text(cards, encoding="utf-8")
    system = ("You are a records clerk at Philip Morris USA filing scanned documents into the company's "
              "records taxonomy. Use only the evidence given. Answer in JSON.\n\n" + cards)

    folds = pd.read_csv(FOLDS, dtype={"id": str})
    routed = pd.read_csv(ROUTED, dtype={"id": str})
    queue = routed.loc[routed["routed"] == "review", "id"].tolist()
    if args.limit:
        queue = queue[: args.limit]
    truth = dict(zip(folds["id"], folds["department"]))
    neighbors = pd.read_csv(RAG / "neighbors.csv", dtype={"id": str, "neighbor_id": str})
    labels = pd.read_csv(LABELS, dtype=str).sort_values("saved_at").drop_duplicates("doc_id", keep="last")
    label_of = dict(zip(labels["doc_id"], labels["type_id"]))
    print("Fitting the classifier for its choices (same model as the routing)...", flush=True)
    choices = classifier_choices(folds)

    out_path = RAG / f"suggestions_{args.variant}.jsonl"
    done = done_ids(out_path)
    todo = [i for i in queue if i not in done]
    print(f"Variant {args.variant}: {len(queue)} review pages, {len(done)} done, {len(todo)} to do", flush=True)

    type_ids = sorted(types)
    progress = Progress(len(todo))
    with out_path.open("a", encoding="utf-8") as f:
        for n, doc_id in enumerate(todo, 1):
            nb = neighbors[neighbors["id"] == doc_id].sort_values("rank")
            t0 = time.time()
            try:
                ans = ask(system, build_prompt(doc_id, nb, label_of, types, choices, args.variant), type_ids)
            except Exception as e:
                print(f"{doc_id}: failed ({str(e)[:120]})", flush=True)
                continue
            rec = {"id": doc_id, "true": truth[doc_id], "variant": args.variant,
                   "department": department_of(ans["record_type"]),
                   "second_department": department_of(ans["second_record_type"]),
                   "classifier_first": choices[doc_id][0][0], "classifier_second": choices[doc_id][1][0],
                   "seconds": round(time.time() - t0, 2), **ans}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            progress(n)

    rows = [json.loads(l) for l in out_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    df = pd.DataFrame(rows)
    df["top1"] = df["department"] == df["true"]
    df["top2"] = df["top1"] | (df["second_department"] == df["true"])
    df["cls_top1"] = df["classifier_first"] == df["true"]
    df["cls_top2"] = df["cls_top1"] | (df["classifier_second"] == df["true"])
    df["cites_source"] = df["sources"].map(lambda s: any(str(x).upper().startswith(("S", "R")) for x in s))
    summary = pd.DataFrame([{
        "variant": args.variant, "pages": len(df),
        "classifier_top1": round(df["cls_top1"].mean(), 3), "rag_top1": round(df["top1"].mean(), 3),
        "classifier_top2": round(df["cls_top2"].mean(), 3), "rag_top2": round(df["top2"].mean(), 3),
        "agrees_with_classifier_first": round((df["department"] == df["classifier_first"]).mean(), 3),
        "cites_a_source": round(df["cites_source"].mean(), 3),
        "seconds_per_page": round(df["seconds"].mean(), 2)}])
    summary.to_csv(RAG / f"rag_summary_{args.variant}.csv", index=False)
    print("\n" + summary.to_string(index=False))
    if args.variant == "A" and len(df) == len(routed[routed["routed"] == "review"]):
        print("\nSuccess rule (set before results): rag_top1 > classifier_top1 on the review queue")
    print(f"Saved to {RAG}")


if __name__ == "__main__":
    main()
