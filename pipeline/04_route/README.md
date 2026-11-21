# 04 Route and suggest

Decides which pages the system files on its own and which go to a person, then tests whether similar filed pages and a local language model help the person reviewing.

A page is filed automatically only when the classifier's calibrated confidence reaches the cutoff chosen for 90% expected accuracy; on the pooled cross-validated predictions that cutoff is 0.55 (112 of 250 pages, 90.2% right). Pages with fewer than 20 tokens of text always go to a person.

Site chapter: **Classification, Routing and Validation** (routing and review helper sections).

| Script | What it does | Reads | Writes |
|---|---|---|---|
| `review_routing.py` | Calibration and the routing curve, threshold chosen on training folds only (TF-IDF); `train_embeddings.py` uses the same rules | folds, cleaned text | `results/classifier/routing_*.csv`, `calibration_summary.csv` |
| `rag_retrieve.py` | Do the five most similar filed pages point to the right department? | folds, embeddings, routed pages | `results/rag/neighbors.csv`, `retrieval_*.csv` |
| `rag_suggest.py` | qwen3:8b suggests a record type for each review page, with (A) and without (C) the classifier's choices | neighbours, labels, cleaned text, taxonomy | `results/rag/suggestions_<A\|C>.jsonl`, `rag_summary_<A\|C>.csv` |
| `rag_autofiled.py` | The same suggestion for the pages that filed themselves, for the app's archive | as above, scale-run results | `results/rag/suggestions_A_autofiled.jsonl`, `results/scale_run/suggestions_A_autofiled.jsonl` |

Commands, from the repository root (Ollama running with `qwen3:8b` for the suggestions):

```powershell
.venv\Scripts\python pipeline\04_route\review_routing.py
.venv\Scripts\python pipeline\04_route\rag_retrieve.py
.venv\Scripts\python pipeline\04_route\rag_suggest.py --variant A
.venv\Scripts\python pipeline\04_route\rag_suggest.py --variant C
.venv\Scripts\python pipeline\04_route\rag_autofiled.py
```

The suggestion scripts save answers as they come and skip pages already answered; delete a `suggestions_*.jsonl` file to run it again. The model runs at temperature 0 with a fixed seed.
