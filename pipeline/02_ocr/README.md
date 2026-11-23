# 02 Reading the pages (OCR)

Reads the first page of every document with five text sources, compares them, and builds the text options the classifier is compared on. PaddleOCR (engine C) was chosen as the main engine.

Site chapter: **Reading the pages**.

| Engine | Script | Writes |
|---|---|---|
| A Tesseract, 150 dpi | `run_ocr.py` | `data/pm_sample/ocr/` |
| B Tesseract, scan resolution | `run_ocr_native.py` | `data/pm_sample/ocr_tess_native/` |
| C PaddleOCR (runs in `.venv-ocr`) | `run_ocr_paddle.py` | `data/pm_sample/ocr_paddle/` |
| D GLM-OCR through Ollama | `run_ocr_vlm.py` | `data/pm_sample/ocr_vlm/` |
| E the archive's own text layer (reference) | `extract_text_layer.py` | `data/pm_sample/ocr_ucsf/` |

| Script | What it does | Reads | Writes |
|---|---|---|---|
| `profile_pages.py` | Measures scan quality from each image and groups the pages (k-means) | page images | `data/pm_sample/page_profile.csv` |
| `name_page_groups.py` | Names the groups and splits group 3 by a speckle rule | page profile | page profile (group, group_name) |
| `check_k_groups.py` | Elbow and silhouette check for k = 2 to 8 | page profile | `results/page_groups_k_check.csv` |
| `extend_test_pages.py` | Extends the test pages to cover every scan group (22 pages) | page profile, test pages | `results/ocr_spotcheck_pm_pages.csv` |
| `ocr_review.py` | Blind review app: every engine's reading side by side, rated and ranked | test pages, images, engine texts | `results/ocr_review_ratings.csv`, `results/scan_group_check.csv` |
| `get_funsd.py`, `eval_ocr_funsd.py` | Accuracy against FUNSD's known text (engines A, C, D) | FUNSD forms | `results/funsd_ocr_eval*.csv` |
| `ocr_eda.py` | Per-engine analysis of all 1,400 pages | engine texts, page profile, labels | `results/ocr_eda/<engine>/` |
| `ocr_compare.py` | The comparison tables the engine choice is made from | all of the above | `results/ocr_compare.csv`, `results/ocr_compare/` |
| `build_ocr_texts.py` | The four candidate texts: C, hybrid, both, D | engines C and D, page profile | `data/pm_sample/ocr_text/` |

Commands, from the repository root (Tesseract 5 for A and B, `ollama pull glm-ocr` for D):

```powershell
.venv\Scripts\python -m pipeline.02_ocr.run_ocr --limit 5
.venv-ocr\Scripts\python -m pipeline.02_ocr.run_ocr_paddle --limit 5
.venv\Scripts\python -m pipeline.02_ocr.ocr_eda --engine C
.venv\Scripts\python -m pipeline.02_ocr.ocr_compare
.venv\Scripts\python -m pipeline.02_ocr.build_ocr_texts
.venv\Scripts\python -m streamlit run pipeline\02_ocr\ocr_review.py
```

The OCR text of every engine is included, so the later steps run without re-reading the pages. The OCR runs need the page images (`pipeline\01_sample\download_pdfs.py`, and for engine A the 150-dpi pages from `get_pm_sample.py`).

FUNSD is not included: `get_funsd.py` downloads it (https://guillaumejaume.github.io/FUNSD/). Only the result tables are in `results/`.

The first OCR pilot, on RVL-CDIP pages, was replaced by the Philip Morris sample; its scripts and data are not included. Two of its result tables are kept in `results/rvl_pilot/` because the site still shows them.
