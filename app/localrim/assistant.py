"""Archive search and question answering, shared by the app's Ask boxes.

Search embeds the question with the same model as the pages (Qwen3-Embedding-0.6B) and
ranks the filed records by similarity. An exact file ID match goes to the top; a phrase
found in a title or on the page, a document year named in the question, and a record type
named in the question each raise the score. Answers come from qwen3:8b through the local
Ollama server at temperature 0, using only the records found, with every claim cited [n].
"""

from __future__ import annotations

import re
from functools import lru_cache

import numpy as np
import pandas as pd

from pipeline.common import ROOT

EMB = ROOT / "results" / "embeddings"
OLLAMA = "http://localhost:11434/api/chat"
LLM = "qwen3:8b"             # same model as the evaluation
EMBED_MODEL = "Qwen/Qwen3-Embedding-0.6B"
TOP_K = 8                    # records handed to the model
DOC_CHARS = 1500             # text per record in the prompt
PHRASE_BOOST, YEAR_BOOST, TYPE_BOOST, EXACT_BOOST = 0.15, 0.2, 0.15, 1.0
GENERIC = {"record", "report", "other", "general", "with", "about", "their", "letter", "note", "plan", "and"}
DECLINE = "does not seem to hold"
ASK_SYSTEM = (
    "You are the records assistant for Philip Morris USA's archive. Answer the user's question using ONLY "
    "the numbered records provided. Cite every claim with the record number in square brackets, like [1] or [2][3]. "
    "Prefer the document date when talking about time. If the records do not answer the question, say plainly "
    "that the archive does not seem to hold the answer, and do not guess. Answer in 2 to 5 sentences, plain prose."
)

_model = None
_vectors = None


@lru_cache(maxsize=4096)
def ocr_text(path: str) -> str:
    """Return a page's OCR text (cached), or an empty string if the file is missing."""
    p = ROOT / path
    return p.read_text(encoding="utf-8", errors="replace") if p.exists() else ""


def embedder():
    """Load the embedding model once, in half precision on a GPU."""
    global _model
    if _model is None:
        import torch
        from sentence_transformers import SentenceTransformer
        device = "cuda" if torch.cuda.is_available() else "cpu"
        _model = SentenceTransformer(EMBED_MODEL, device=device)
        if device == "cuda":
            _model.half()
        _model.max_seq_length = 2048
    return _model


def page_vectors():
    """Return (ids, vectors) for the saved pages plus the pages added in the app."""
    global _vectors
    if _vectors is None:
        ids = pd.read_csv(EMB / "hybrid_qwen3-0.6b_ids.csv", dtype={"id": str})["id"].tolist()
        vecs = np.load(EMB / "hybrid_qwen3-0.6b.npy")
        new_ids, new_vecs = ROOT / "outputs" / "app" / "new_vector_ids.csv", ROOT / "outputs" / "app" / "new_vectors.npy"
        if new_ids.exists() and new_vecs.exists():
            ids += pd.read_csv(new_ids, dtype=str)["id"].tolist()
            vecs = np.vstack([vecs, np.load(new_vecs).astype(vecs.dtype)])
        _vectors = (ids, vecs)
    return _vectors


def reset_vectors() -> None:
    """Make page_vectors() reload after pages are added."""
    global _vectors
    _vectors = None


def embed_query(q: str) -> np.ndarray:
    """Embed a question with the model's query prompt."""
    m = embedder()
    try:
        v = m.encode([q], prompt_name="query", normalize_embeddings=True, convert_to_numpy=True)
    except Exception:
        v = m.encode([q], normalize_embeddings=True, convert_to_numpy=True)
    return v[0].astype(np.float32)


def type_words(type_names: dict) -> dict:
    """Map record type ID -> the words of its name, for spotting types named in a question."""
    return {t: {w.rstrip("s") for w in re.findall(r"[a-z]{4,}", name.lower())} - GENERIC
            for t, name in type_names.items()}


def retrieve(q: str, archive: pd.DataFrame, ids: list[str], types: dict, k: int = TOP_K) -> pd.DataFrame:
    """Return the k best records for q among ids, best first, with a score column."""
    all_ids, vecs = page_vectors()
    row = {d: i for i, d in enumerate(all_ids)}
    pool = archive[archive["id"].isin(ids) & archive["id"].isin(list(row))].copy()
    if pool.empty:
        return pool
    pool["score"] = vecs[[row[i] for i in pool["id"]]] @ embed_query(q)
    ql = q.strip().lower()
    exact = (pool["file_id"].str.lower() == ql) | (pool["bates"].str.lower() == ql)
    phrase = pool["title"].str.lower().str.contains(ql, regex=False) | pool["text"].map(lambda p: ql in ocr_text(p).lower())
    pool.loc[phrase, "score"] += PHRASE_BOOST
    years = set(re.findall(r"\b(1[89]\d\d|20\d\d)\b", q))
    if years:
        pool.loc[pool["date_iso"].str[:4].isin(years), "score"] += YEAR_BOOST
    words = {w.rstrip("s") for w in re.findall(r"[a-z]{4,}", ql)}
    hit = [t for t, kw in types.items() if kw & words]
    if hit:
        pool.loc[pool["record_type"].isin(hit), "score"] += TYPE_BOOST
    pool.loc[exact, "score"] += EXACT_BOOST
    return pool.sort_values("score", ascending=False).head(k).reset_index(drop=True)


def ask_llm(q: str, docs: pd.DataFrame) -> str | None:
    """Answer q from docs with the local model; None if Ollama can't be reached."""
    import requests
    parts = []
    for n, d in enumerate(docs.itertuples(), 1):
        text = " ".join(ocr_text(d.text).split())[:DOC_CHARS] or "(no readable text)"
        filing = d.department_name + (f" › {d.series_name} › {d.record_type_name}" if d.record_type else "")
        parts.append(f"[{n}] {d.title} | file {d.file_id} | document date {d.date_shown} | filed under {filing}\n{text}")
    prompt = "RECORDS:\n\n" + "\n\n".join(parts) + f"\n\nQUESTION: {q}"
    try:
        r = requests.post(OLLAMA, timeout=180, json={
            "model": LLM, "stream": False, "think": False,
            "options": {"temperature": 0, "seed": 44, "num_ctx": 8192},
            "messages": [{"role": "system", "content": ASK_SYSTEM}, {"role": "user", "content": prompt}]})
        r.raise_for_status()
        return r.json()["message"]["content"].strip()
    except Exception:
        return None


def cited(answer: str | None, n_docs: int) -> list[int]:
    """Return the record numbers cited in the answer, limited to 1..n_docs."""
    return sorted({int(n) for n in re.findall(r"\[(\d+)\]", answer or "") if 0 < int(n) <= n_docs})
