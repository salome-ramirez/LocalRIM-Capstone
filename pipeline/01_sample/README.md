# 01 Sample and labels

Draws a random sample of 1,400 public, scanned Philip Morris documents from the UCSF Industry Documents Library, downloads them, and labels the first page of each training and test document by hand.

Site chapters: **Filing system** (the sample, the labels and the pilot rounds) and **Project log**.

| Script | What it does | Reads | Writes |
|---|---|---|---|
| `get_pm_sample.py` | Draws the documents at random (seed 44) and saves the manifest | the library's search API | `data/pm_sample/manifest.csv`, `data/pm_sample/images/` |
| `repull_pm_sample.py` | Downloads every sampled document again in full: PDF, all metadata fields, first page at scan resolution | manifest | `data/pm_sample/pdf/`, `images_native/`, `metadata_full.csv`, `pdf_info.csv` |
| `download_pdfs.py` | Downloads the PDFs and renders the first pages for a fresh copy of this repository | `data/pm_sample/pdf_info.csv` | `data/pm_sample/pdf/`, `images_native/` |
| `labeler.py` | Labeling app: department, series, record type, "also fits", certainty, or exclusion | manifest, page images, taxonomy | `labels/label_pm_labels_<annotator>.csv`, `labels/final_test_labels_<annotator>.csv` |
| `draw_final_test.py` | Draws the 150-label blind final test from the scale batch (seed 44) | manifest | `results/final_test_pages.csv` |

Commands, from the repository root:

```powershell
.venv\Scripts\python pipeline\01_sample\download_pdfs.py
.venv\Scripts\streamlit run pipeline\01_sample\labeler.py
```

`get_pm_sample.py`, `repull_pm_sample.py` and `draw_final_test.py` record how the sample was made. Running them again draws or downloads into the same files; their saved outputs are already in the repository, so you don't need to.

The labels in `labels/` are the ones the results were computed from. Free-text label notes are not published: the `note` column keeps only the exclusion reason of excluded pages and the marker "Non-English", the two things the counts read.
