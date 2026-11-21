"""Embed every page's text once and save the vectors.

Uses Qwen3-Embedding-0.6B (sentence-transformers, local GPU): small, multilingual and
with a long context. Pages are cut at MAX_TOKENS tokens and the number cut is reported.
All 1,400 pages are embedded; no labels are used.

--text picks the input: hybrid (default; the classifier's text, own Bates removed),
hybrid_with_bates (Bates left in, for the ablation) or A (engine A's text).

Reads data/pm_sample/ocr_text_clean/<text>/<id>.txt (hybrid_with_bates: ocr_text/hybrid).
Writes results/embeddings/<text>_qwen3-0.6b.npy (normalised vectors, one row per page)
and <text>_qwen3-0.6b_ids.csv (row order: id, characters, tokens, cut).

Usage:
    python pipeline/03_classify/embed_pages.py
    python pipeline/03_classify/embed_pages.py --text hybrid_with_bates
    python pipeline/03_classify/embed_pages.py --text A
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer

import sys
sys.path[:0] = [str(Path(__file__).resolve().parents[2] / "pipeline" / step) for step in ("02_ocr",)]  # sibling pipeline steps
from ocr_eda import NOT_RETRIEVED

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "data" / "pm_sample" / "manifest.csv"
TEXT_DIRS = {
    "hybrid": ROOT / "data" / "pm_sample" / "ocr_text_clean" / "hybrid",
    "hybrid_with_bates": ROOT / "data" / "pm_sample" / "ocr_text" / "hybrid",
    "A": ROOT / "data" / "pm_sample" / "ocr_text_clean" / "A",
}
OUT = ROOT / "results" / "embeddings"

MODEL_NAME = "Qwen/Qwen3-Embedding-0.6B"
MODEL_TAG = "qwen3-0.6b"
MAX_TOKENS = 2048
BATCH_SIZE = 8


def main() -> None:
    """Embed every page of the chosen text and save the vectors and row order."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", choices=list(TEXT_DIRS), default="hybrid")
    text_name = parser.parse_args().text
    TEXTS = TEXT_DIRS[text_name]
    RUN_NAME = f"{text_name}_{MODEL_TAG}"
    print(f"Text: {text_name} ({TEXTS})")

    manifest = pd.read_csv(MANIFEST, dtype=str)
    ids = [i for i in manifest["id"] if i not in NOT_RETRIEVED]
    texts = [(TEXTS / f"{i}.txt").read_text(encoding="utf-8", errors="replace").strip()
             if (TEXTS / f"{i}.txt").exists() else "" for i in ids]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading {MODEL_NAME} on {device} (first run downloads about 1.2 GB)...", flush=True)
    model = SentenceTransformer(MODEL_NAME, device=device)
    if device == "cuda":
        model.half()
    model.max_seq_length = MAX_TOKENS

    token_counts = [len(model.tokenizer(t, add_special_tokens=True)["input_ids"]) for t in texts]

    start = time.time()
    vectors = model.encode(texts, batch_size=BATCH_SIZE, normalize_embeddings=True,
                           show_progress_bar=True, convert_to_numpy=True)
    seconds = time.time() - start

    OUT.mkdir(parents=True, exist_ok=True)
    np.save(OUT / f"{RUN_NAME}.npy", vectors.astype(np.float32))
    info = pd.DataFrame({
        "id": ids,
        "characters": [len(t) for t in texts],
        "tokens": token_counts,
        "cut": [n > MAX_TOKENS for n in token_counts],
    })
    info.to_csv(OUT / f"{RUN_NAME}_ids.csv", index=False)

    print(f"\nPages: {len(ids)} | empty: {sum(1 for t in texts if not t)} | "
          f"cut at {MAX_TOKENS} tokens: {int(info['cut'].sum())}")
    print(f"Vector size: {vectors.shape[1]} | time: {seconds:.0f} s "
          f"({seconds / len(ids):.2f} s per page)")
    print(f"Saved to {OUT}")


if __name__ == "__main__":
    main()
