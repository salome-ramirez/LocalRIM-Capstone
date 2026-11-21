"""Intake pipeline: one page in, one filing decision out.

  read the page with PaddleOCR (in .venv-ocr, through localrim.ocr_worker)
  -> find the page's own file number (Bates): kept with the record, removed from the
     text the classifier sees
  -> department and confidence from the classifier
  -> route: auto-filed, or to a person with the reason
  -> suggested record type with a reason and sources, from the local language model

The settings are imported from the evaluation modules, not copied: classifier
(train_embeddings), cutoff and minimum tokens (final_test), footer and file-number rules
(run_ocr_paddle, ocr_eda, prep_classifier_text), taxonomy cards, model call and prompt
wording (rag_suggest). Two differences from the evaluation: the page is read by PaddleOCR
alone, and its file number is read from the page instead of looked up in the manifest.

Runs in .venv (Python 3.14). Needs Ollama running for the record-type suggestion.

Usage (from the app folder):
    python -m localrim.pipeline --check 5            # 5 scale-run pages, compared with the saved results
    python -m localrim.pipeline --check 5 --no-rag   # same, without the language model
    python -m localrim.pipeline --file-numbers 50    # file-number detection on 50 pages vs the manifest
    python -m localrim.pipeline path\\to\\page.png    # one page of your own (PNG, or a PDF's first page)
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path[:0] = [str(Path(__file__).resolve().parents[2] / "pipeline" / step) for step in ("02_ocr", "03_classify", "04_route", "05_validate",)]  # sibling pipeline steps
from final_test import CUTOFF, K, MIN_TOKENS, TRAIN_LABELS, latest
from ocr_eda import PUNCTUATION, STRAIGHT, UPSIDE_DOWN, is_bates_like, matches_bates
from prep_classifier_text import remove_own_bates
from rag_suggest import (NEIGHBOR_CHARS, PAGE_CHARS, TAXONOMY, ask, build_cards, department_of,
                                  read_text)
from train_embeddings import EMB_DIR, MODEL_TAG, calibrated_model, load_vectors

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "pm_sample"
FOLDS = ROOT / "results" / "cv_folds.csv"
OCR_PYTHON = ROOT / ".venv-ocr" / "Scripts" / "python.exe"
OCR_WORKER = "localrim.ocr_worker"
OUT = ROOT / "outputs" / "app"
EMBED_MODEL = "Qwen/Qwen3-Embedding-0.6B"   # same model as the saved page embeddings
EMBED_MAX_TOKENS = 2048                      # same limit as the saved page embeddings
PDF_DPI = 300                                # only when a PDF page is not a single scanned image
PREFIX = "@@RESULT "
SYSTEM_INTRO = ("You are a records clerk at Philip Morris USA filing scanned documents into the company's "
                "records taxonomy. Use only the evidence given. Answer in JSON.\n\n")


# Reading the page

class OcrWorker:
    """Keeps localrim.ocr_worker running in .venv-ocr so PaddleOCR loads once."""

    def __init__(self):
        self.proc = subprocess.Popen([str(OCR_PYTHON), "-m", OCR_WORKER, "--serve"], cwd=ROOT / "app",
                                     stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                                     text=True, encoding="utf-8", bufsize=1)
        self._next()  # wait until the models are loaded

    def _next(self) -> dict:
        for line in self.proc.stdout:
            if line.startswith(PREFIX):
                return json.loads(line[len(PREFIX):])
        raise RuntimeError("The OCR helper stopped. Check that .venv-ocr exists and PaddleOCR runs there.")

    def read(self, image: Path) -> dict:
        """Read one page image and return the OCR result."""
        self.proc.stdin.write(json.dumps({"path": str(image)}) + "\n")
        self.proc.stdin.flush()
        return self._next()

    def close(self):
        """Stop the helper process."""
        if self.proc.poll() is None:
            self.proc.stdin.close()
            self.proc.wait(timeout=10)


def page_image(path: Path) -> Path:
    """Return a PNG of the document's first page.

    A PDF page that is one scanned image is saved at its native resolution, as the
    evaluation read it; otherwise it is rendered at PDF_DPI.
    """
    if path.suffix.lower() != ".pdf":
        return path
    import fitz
    OUT.joinpath("pages").mkdir(parents=True, exist_ok=True)
    target = OUT / "pages" / f"{path.stem}.png"
    with fitz.open(path) as doc:
        page = doc[0]
        images = page.get_images(full=True)
        if len(images) == 1:
            pix = fitz.Pixmap(doc, images[0][0])
            if pix.n - pix.alpha >= 4:
                pix = fitz.Pixmap(fitz.csRGB, pix)
        else:
            pix = page.get_pixmap(dpi=PDF_DPI)
        pix.save(target)
    return target


# File number (Bates) found on the page

def find_file_number(items: list[dict]) -> dict:
    """Return the page's own file number as read, and the raw token it came from.

    Uses the evaluation's rule for what a Bates number looks like (ocr_eda.is_bates_like)
    and its digit fixes for misread or upside-down stamps; among candidates, the lowest on
    the page wins. Returns empty strings when none is found.
    """
    best = None
    for r in items:
        for token in r["text"].split():
            if not is_bates_like(token):
                continue
            t = token.strip(PUNCTUATION)
            for cand in (t.translate(STRAIGHT), t[::-1].translate(UPSIDE_DOWN)):
                core = re.match(r"^(\d{9,11})([A-Z]?)$", cand)
                if core:
                    y = (r["y1"] + r["y2"]) / 2
                    if best is None or y > best[0]:
                        best = (y, core.group(0), token)
                    break
    return {"file_number": best[1], "file_number_raw": best[2]} if best else {"file_number": "", "file_number_raw": ""}


# The pipeline

class Pipeline:
    """Load the OCR helper, embedding model, classifier and taxonomy once; file() handles one page."""
    def __init__(self, rag: bool = True):
        t0 = time.time()
        self.ocr = OcrWorker()
        self.ocr_load_seconds = round(time.time() - t0, 1)

        import torch
        from sentence_transformers import SentenceTransformer
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.embedder = SentenceTransformer(EMBED_MODEL, device=device)
        if device == "cuda":
            self.embedder.half()
        self.embedder.max_seq_length = EMBED_MAX_TOKENS

        # Classifier fitted exactly as in the final test
        train = pd.read_csv(FOLDS, dtype={"id": str}).reset_index(drop=True)
        self.train_ids = train["id"].to_numpy()
        self.X_train = load_vectors(train["id"], "hybrid")
        self.y_train = train["department"].to_numpy()
        self.model = calibrated_model().fit(self.X_train, self.y_train)

        self.rag = rag
        cards, self.types = build_cards(yaml.safe_load(TAXONOMY.read_text(encoding="utf-8")))
        self.system = SYSTEM_INTRO + cards
        labels = latest(TRAIN_LABELS)
        self.label_of = dict(zip(labels["doc_id"], labels["type_id"]))
        self.load_seconds = round(time.time() - t0, 1)

    def prompt(self, page_text: str, neighbors: list[dict], choices) -> str:
        """Return rag_suggest.build_prompt (variant A) word for word, with the page's own text."""
        flat = " ".join(page_text.split())
        parts = ["PAGE TO FILE (first page of a scanned Philip Morris document; OCR text):",
                 flat[:PAGE_CHARS] if flat else "(no readable text)", "",
                 "SIMILAR PAGES ALREADY FILED (most similar first):"]
        for k, nb in enumerate(neighbors, 1):
            t = self.label_of.get(nb["id"], "")
            parts.append(f"S{k}. Filed as {t} {self.types.get(t, '')}. Similarity {nb['similarity']:.2f}. "
                         f"Text: {read_text(nb['id'], NEIGHBOR_CHARS)}")
        (d1, p1), (d2, p2) = choices
        parts += ["", f"CLASSIFIER (department level; Administration counted with Legal): "
                      f"first choice {d1} ({p1:.0%} confident), second choice {d2} ({p2:.0%})."]
        parts += ["", "Which Philip Morris department would keep the official copy of this page, and which "
                      "record type is it? Decide by the page's purpose, using the cards and rules. Give your "
                      "best record type, a second choice from a DIFFERENT department, a one-line reason, and "
                      "the sources you relied on (S1-S5, rule IDs such as R9, CLASSIFIER)."]
        return "\n".join(parts)

    def file(self, path: Path) -> dict:
        """Read, classify and route one page; with rag, also suggest a record type."""
        times = {}
        t0 = time.time()
        image = page_image(Path(path))
        read = self.ocr.read(image)
        times["read"] = round(time.time() - t0, 2)
        if read.get("error"):
            return {"image": str(image), "error": read["error"]}

        fn = find_file_number(read["items"])
        clean, removed = remove_own_bates(read["text"], fn["file_number"]) if fn["file_number"] else (read["text"], [])

        t0 = time.time()
        tokens = len(self.embedder.tokenizer(clean, add_special_tokens=True)["input_ids"])
        vec = self.embedder.encode([clean], normalize_embeddings=True, convert_to_numpy=True).astype(np.float32)
        proba = self.model.predict_proba(vec)[0]
        order = np.argsort(-proba)
        first, second = self.model.classes_[order[0]], self.model.classes_[order[1]]
        conf, conf2 = float(proba[order[0]]), float(proba[order[1]])
        times["classify"] = round(time.time() - t0, 2)

        if not clean.strip():
            reason = "no readable text"
        elif tokens < MIN_TOKENS:
            reason = "under 20 tokens"
        elif conf >= CUTOFF:
            reason = "confident"
        else:
            reason = "below cutoff"
        routed = "auto-filed" if reason == "confident" else "review"

        sims = (vec @ self.X_train.T)[0]
        neighbors = [{"id": self.train_ids[j], "department": self.y_train[j], "similarity": round(float(sims[j]), 4),
                      "record_type": self.label_of.get(self.train_ids[j], "")} for j in np.argsort(-sims)[:K]]

        out = {"image": str(image), "text": read["text"], "items": read["items"], "rotated": read["rotated"],
               "words": read["words"], "ocr_confidence": read["mean_conf"], **fn,
               "file_number_removed": removed, "tokens": tokens,
               "department": first, "confidence": round(conf, 3), "second": second, "second_confidence": round(conf2, 3),
               "routed": routed, "route_reason": reason, "neighbors": neighbors, "error": "",
               "vector": vec[0].tolist()}   # kept so the app's assistant can search new records

        if self.rag:
            t0 = time.time()
            try:
                ans = ask(self.system, self.prompt(clean, neighbors, [(first, conf), (second, conf2)]), sorted(self.types))
                out["suggestion"] = {"record_type": ans["record_type"], "record_type_name": self.types.get(ans["record_type"], ""),
                                     "department": department_of(ans["record_type"]),
                                     "second_record_type": ans["second_record_type"],
                                     "reason": ans["reason"], "sources": ans["sources"]}
            except Exception as e:
                out["suggestion"] = {"error": f"Assistant unavailable ({type(e).__name__})"}
            times["suggest"] = round(time.time() - t0, 2)
        out["seconds"] = times
        return out

    def close(self):
        """Stop the OCR helper."""
        self.ocr.close()


# Checks against the saved evaluation results

def scale_pages(n: int) -> list[str]:
    """Return n scale-run pages, every k-th one, so both routes and many departments appear."""
    preds = pd.read_csv(ROOT / "results" / "scale_run" / "predictions.csv", dtype={"id": str})
    have = [i for i in preds["id"] if (DATA / "images_native" / f"{i}.png").exists()]
    step = max(len(have) // n, 1)
    return have[::step][:n]


def check(n: int, rag: bool) -> None:
    """Run n scale-run pages through the pipeline and compare with the saved results."""
    preds = pd.read_csv(ROOT / "results" / "scale_run" / "predictions.csv", dtype={"id": str}).set_index("id")
    saved = {}
    for name in ["suggestions_A.jsonl", "suggestions_A_autofiled.jsonl"]:
        for l in (ROOT / "results" / "scale_run" / name).read_text(encoding="utf-8").splitlines():
            if l.strip():
                r = json.loads(l)
                saved[r["id"]] = r["record_type"]
    bates = pd.read_csv(DATA / "manifest.csv", dtype=str).set_index("id")["bates"]

    print("Loading (OCR helper, embedding model, classifier)...", flush=True)
    p = Pipeline(rag=rag)
    print(f"Loaded in {p.load_seconds} s (OCR helper {p.ocr_load_seconds} s)\n", flush=True)
    rows = []
    try:
        for doc_id in scale_pages(n):
            r = p.file(DATA / "images_native" / f"{doc_id}.png")
            if r.get("error"):
                print(f"{doc_id}: {r['error']}")
                continue
            s = preds.loc[doc_id]
            row = {"id": doc_id,
                   "file_number": r["file_number"], "manifest_bates": bates.get(doc_id, ""),
                   "file_number_ok": matches_bates(r["file_number"], str(bates.get(doc_id, ""))) if r["file_number"] else False,
                   "department": r["department"], "saved_department": s["predicted"],
                   "confidence": r["confidence"], "saved_confidence": s["confidence"],
                   "routed": r["routed"], "saved_routed": s["routed"],
                   **{f"seconds_{k}": v for k, v in r["seconds"].items()}}
            if rag:
                row["record_type"] = r["suggestion"].get("record_type", r["suggestion"].get("error"))
                row["saved_record_type"] = saved.get(doc_id, "")
            rows.append(row)
            print(f"{doc_id}: {r['department']} {r['confidence']:.2f} {r['routed']}"
                  + (f" | {row['record_type']}" if rag else "") + f" | file no. {r['file_number'] or '-'}", flush=True)
    finally:
        p.close()

    df = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / "pipeline_check.csv", index=False)
    pd.set_option("display.width", 200)
    print("\n" + df.drop(columns=[c for c in df.columns if c.startswith("seconds")]).to_string(index=False))
    print(f"\nSame department as saved: {(df['department'] == df['saved_department']).mean():.0%} | "
          f"same route: {(df['routed'] == df['saved_routed']).mean():.0%}"
          + (f" | same record type: {(df['record_type'] == df['saved_record_type']).mean():.0%}" if rag else "")
          + f" | file number matches manifest: {df['file_number_ok'].mean():.0%}")
    print("Differences are expected where PaddleOCR's text differs from the evaluation's hybrid text (App-1).")
    print(f"Saved to {OUT / 'pipeline_check.csv'}")


def check_file_numbers(n: int) -> None:
    """Compare file-number detection on n pages with the manifest."""
    bates = pd.read_csv(DATA / "manifest.csv", dtype=str).set_index("id")["bates"]
    ocr = OcrWorker()
    rows = []
    try:
        for doc_id in scale_pages(n):
            r = ocr.read(DATA / "images_native" / f"{doc_id}.png")
            if r.get("error"):
                continue
            fn = find_file_number(r["items"])
            truth = str(bates.get(doc_id, ""))
            rows.append({"id": doc_id, "found": fn["file_number"], "raw": fn["file_number_raw"], "manifest": truth,
                         "exact": fn["file_number"][:10] == truth[:10] if fn["file_number"] else False,
                         "close": matches_bates(fn["file_number"], truth) if fn["file_number"] else False})
    finally:
        ocr.close()
    df = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / "file_number_check.csv", index=False)
    print(df.to_string(index=False))
    print(f"\n{len(df)} pages | found a number: {(df['found'] != '').mean():.0%} | "
          f"exact (first 10 characters): {df['exact'].mean():.0%} | close: {df['close'].mean():.0%}")
    print(f"Saved to {OUT / 'file_number_check.csv'}")


def main() -> None:
    """Command line: --check, --file-numbers or one page."""
    parser = argparse.ArgumentParser()
    parser.add_argument("page", nargs="?", help="a PNG or PDF of your own")
    parser.add_argument("--check", type=int, help="run this many scale-run pages and compare with the saved results")
    parser.add_argument("--file-numbers", type=int, help="file-number detection on this many pages")
    parser.add_argument("--no-rag", action="store_true", help="skip the LLM suggestion")
    args = parser.parse_args()
    if args.check:
        check(args.check, rag=not args.no_rag)
    elif args.file_numbers:
        check_file_numbers(args.file_numbers)
    elif args.page:
        p = Pipeline(rag=not args.no_rag)
        try:
            r = p.file(Path(args.page))
        finally:
            p.close()
        r.pop("items", None)
        r.pop("vector", None)
        print(json.dumps(r, indent=2, default=str))
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
