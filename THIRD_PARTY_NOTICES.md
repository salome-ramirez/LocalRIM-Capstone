# Third-party notices

LocalRIM's own code is under the MIT licence (see [LICENSE](LICENSE)). It uses the software, models, fonts and data below, each under its own licence. None of them is changed or redistributed here except the two fonts in `app/static/fonts/` and the site's fonts in `docs/`.

## Fonts (included)

| Font | Licence | File |
|---|---|---|
| Literata | SIL Open Font License 1.1 | `app/static/fonts/OFL-literata.txt` |
| Atkinson Hyperlegible Next | SIL Open Font License 1.1 | `app/static/fonts/OFL-atkinson-hyperlegible-next.txt` |

## Models (downloaded at setup or first use)

| Model | Used for | Licence |
|---|---|---|
| Qwen3-Embedding-0.6B (Hugging Face) | page and question embeddings | Apache 2.0 |
| Qwen3 8B (`qwen3:8b`, through Ollama) | record-type suggestions, titles and dates, answers | Apache 2.0 |
| PaddleOCR default models (PP-OCRv6 detection and recognition, PP-LCNet orientation) | reading pages | Apache 2.0 |
| GLM-OCR (`glm-ocr`, through Ollama; OCR engine D, only to re-run step 02) | reading pages | see its model page on ollama.com |

## Software (installed from requirements.txt and requirements-ocr.txt)

Main packages and their licences. The full list, with versions, is in the two requirements files; each package's licence is in its installed metadata.

| Package | Licence |
|---|---|
| Streamlit | Apache 2.0 |
| PyTorch | BSD 3-Clause |
| sentence-transformers, transformers, huggingface_hub, datasets | Apache 2.0 |
| scikit-learn, SciPy, NumPy, pandas | BSD 3-Clause |
| matplotlib | Matplotlib licence (PSF-based) |
| PyMuPDF | GNU AGPL 3.0 (or a commercial licence from Artifex) |
| Pillow | MIT-CMU |
| PyYAML | MIT |
| requests | Apache 2.0 |
| pytesseract | Apache 2.0 |
| jiwer | Apache 2.0 |
| PaddlePaddle, PaddleOCR, PaddleX | Apache 2.0 |
| OpenCV | Apache 2.0 |
| Ollama (installed separately) | MIT |
| Tesseract OCR (installed separately; only to re-run step 02) | Apache 2.0 |

PyMuPDF is AGPL-licensed. Running LocalRIM on your own computer is fine; if you distribute the app or offer it as a network service, the AGPL's terms apply to that use.

## Documents and derived data

The documents are from the Philip Morris Records collection of the [UCSF Industry Documents Library](https://www.industrydocuments.ucsf.edu/); see its [Copyright and Fair Use](https://www.industrydocuments.ucsf.edu/about-idl/copyright/) page. They are not included; `pipeline/01_sample/download_pdfs.py` downloads them from the library. The derived data in this repository (catalogue metadata, OCR text, embeddings, labels, classifier results and suggestions) was made from those documents; see [data/README.md](data/README.md).

## Datasets used in the evaluation (not included)

| Dataset | Used for | Where to get it |
|---|---|---|
| FUNSD (199 scanned forms with word-level ground truth) | OCR accuracy of engines A, C and D | https://guillaumejaume.github.io/FUNSD/ (`pipeline/02_ocr/get_funsd.py` downloads it); see its page for terms of use |
| RVL-CDIP | the first OCR pilot, before the Philip Morris sample | https://adamharley.com/rvl-cdip/ ; only two result tables are included (`results/rvl_pilot/`) |
