# 03 Classify

Prepares the classifier's text, fixes the cross-validation folds, and compares a TF-IDF classifier with one on page embeddings. Every model is scored on the same 5 folds of the 250 training labels, by macro F1 at the department level (Administration counted with Legal).

Site chapter: **Classification, Routing and Validation** (classifier section).

| Script | What it does | Reads | Writes |
|---|---|---|---|
| `prep_classifier_text.py` | Removes each page's own Bates number from the text the classifier sees | `data/pm_sample/ocr_text/`, manifest | `data/pm_sample/ocr_text_clean/`, `results/bates_removals.csv` |
| `make_folds.py` | 5 folds, stratified by department and grouped by archive box | training labels, `metadata_full.csv` | `results/cv_folds.csv` |
| `train_tfidf.py` | TF-IDF + logistic regression on every text option | folds, cleaned texts | `results/classifier/tfidf_cv_*.csv` |
| `bates_probe.py` | Can the department be guessed from the Bates number alone? | folds, manifest | `results/classifier/bates_probe.csv` |
| `embed_pages.py` | Qwen3-Embedding-0.6B vectors for every page (GPU) | cleaned texts | `results/embeddings/` |
| `train_embeddings.py` | Calibrated logistic regression on the embeddings, with the 90% routing check | folds, embeddings | `results/classifier/emb_*.csv` |
| `tfidf_errors.py` | Confusion matrix and per-department scores | saved predictions, labels | `results/classifier/confusion_*.csv`, `per_department_*.csv` |

Commands, in this order, from the repository root:

```powershell
.venv\Scripts\python pipeline\03_classify\prep_classifier_text.py
.venv\Scripts\python pipeline\03_classify\make_folds.py
.venv\Scripts\python pipeline\03_classify\train_tfidf.py
.venv\Scripts\python pipeline\03_classify\bates_probe.py
.venv\Scripts\python pipeline\03_classify\embed_pages.py
.venv\Scripts\python pipeline\03_classify\embed_pages.py --text hybrid_with_bates
.venv\Scripts\python pipeline\03_classify\embed_pages.py --text A
.venv\Scripts\python pipeline\03_classify\train_embeddings.py
.venv\Scripts\python pipeline\03_classify\train_embeddings.py --text hybrid_with_bates
.venv\Scripts\python pipeline\03_classify\train_embeddings.py --text A
.venv\Scripts\python pipeline\03_classify\tfidf_errors.py
.venv\Scripts\python pipeline\03_classify\tfidf_errors.py --embeddings
```

Headline results: cross-validated macro F1 0.683 with embeddings and 0.563 with TF-IDF (hybrid text).
