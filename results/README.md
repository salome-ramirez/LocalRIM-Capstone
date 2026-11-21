# Results

The saved outputs of every pipeline step, as tables. No page images.

| Folder or file | Written by |
|---|---|
| `final_test_pages.csv` | `pipeline/01_sample/draw_final_test.py` |
| `ocr_spotcheck_pm*.csv`, `ocr_review_ratings.csv`, `scan_group_check.csv`, `page_groups_k_check.csv`, `funsd_ocr_*.csv`, `ocr_eda/`, `ocr_compare.csv`, `ocr_compare/` | `pipeline/02_ocr/` |
| `rvl_pilot/` | the first OCR pilot on RVL-CDIP pages (scripts not included; kept because the site shows it) |
| `bates_removals.csv`, `cv_folds.csv`, `embeddings/`, `classifier/` | `pipeline/03_classify/` (and the routing tables from `pipeline/04_route/review_routing.py`) |
| `rag/` | `pipeline/04_route/` |
| `final_test/`, `scale_run/`, `record_types/`, `archive_check/` | `pipeline/05_validate/` |
| `batch_metrics.csv` | `pipeline/06_site_data/` |

Free-text notes in the rating tables are blanked. The language model's one-line reasons are kept, except for the five pages withheld for personal data (see `data/README.md`).
