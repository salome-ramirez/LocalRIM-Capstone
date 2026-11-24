# LocalRIM

LocalRIM files scanned business documents into a records taxonomy (department, record series, record type) on one Windows PC. It reads each page with OCR, classifies it with page embeddings and a calibrated classifier, files a page on its own only when it is confident enough, and sends the rest to a person; a local language model suggests the record type and answers questions about the archive, citing the records it used. Documents, searches and questions never leave the computer.

This repository reproduces the whole project on a random sample of public documents from the Philip Morris Records collection of the UCSF Industry Documents Library, with the taxonomy and labels made for it. Another organization would need its own taxonomy and labels; adapting the pipeline to them is future work.

## Example pages

Seven first pages from the labeled sample, one for each department in the taxonomy, with the record type each was filed under.

| Page | Department | Record type | Why it is filed there |
|---|---|---|---|
| [rhhn0043](examples/rhhn0043.png) | Marketing and sales | Ads and ad proofs | Philip Morris's own 1997 menthol print ad |
| [qrkw0107](examples/qrkw0107.png) | Research and development | Study and lab reports | Monthly progress on experimental tobacco blends, 1982 |
| [gsnw0104](examples/gsnw0104.png) | Quality, safety and health | Specifications and standards | A leaf grade standard with targets and control limits |
| [htng0175](examples/htng0175.png) | Corporate affairs | Legislation tracking and advocacy | Draft testimony against a state restaurant smoking ban |
| [qrml0111](examples/qrml0111.png) | Legal | Patents and trademarks | Decisions on foreign patent filings |
| [fyvv0020](examples/fyvv0020.png) | Finance | Invoices | A freight invoice for a billboard job: invoices go to Finance whatever the activity |
| [szpk0065](examples/szpk0065.png) | Administration and records (scored with Legal) | Records management | Philip Morris's own blank records disposal form |

## The steps

| Step | What it does | Site chapter |
|---|---|---|
| [01 Sample and labels](pipeline/01_sample/) | Draw and download 1,400 documents; label the training and final-test pages | Filing system |
| [02 Reading the pages](pipeline/02_ocr/) | Five text sources compared; PaddleOCR chosen; the classifier's text options built | Reading the pages |
| [03 Classify](pipeline/03_classify/) | Cross-validated TF-IDF and embedding classifiers (macro F1 0.563 and 0.683) | Classification, Routing and Validation |
| [04 Route and suggest](pipeline/04_route/) | File automatically only at 90% expected accuracy (cutoff 0.55); record-type suggestions for the pages a person reviews | Classification, Routing and Validation |
| [05 Validate](pipeline/05_validate/) | One-time final test on 150 unseen pages (macro F1 0.60; 41% filed automatically, 54 of 62 right); 1,000-document scale run; archive assistant check | Findings, The app |
| [06 Site data](pipeline/06_site_data/) | Every number the site shows, in `docs/data/site_data.json` | all |
| [app](app/) | The LocalRIM application: Archive, Intake and Review queue | The app |

Also in the repository: [taxonomy/](taxonomy/) (departments, series, record types and the labeling rules), [labels/](labels/), [data/](data/) (document list, OCR text, derived files), [results/](results/) (every step's saved tables) and [docs/](docs/) (the project site).

## Requirements

- Windows 10 or 11
- An NVIDIA GPU with a current driver (developed on an RTX 4060 Ti, 8 GB)
- Python 3.14 for the app and the pipeline, and Python 3.12 for PaddleOCR, both from [python.org](https://www.python.org/downloads/windows/), with the `py` launcher
- [Ollama](https://ollama.com/download) with `qwen3:8b`
- About 15 GB of free disk space: the two environments, the models and the documents
- Only to re-run the OCR step: [Tesseract 5](https://github.com/UB-Mannheim/tesseract/wiki) with the English and German language data (engines A and B), and `ollama pull glm-ocr` (engine D)

## Setup

Run these in PowerShell from the repository folder.

1. Create the main environment (Python 3.14) and install its packages. `requirements.txt` points pip to the PyTorch CUDA 12.8 index for `torch`.

   ```powershell
   py -3.14 -m venv .venv
   .venv\Scripts\python -m pip install -r requirements.txt
   ```

2. Create the OCR environment (Python 3.12). PaddleOCR is not available for Python 3.14; the app starts it in this environment when it reads new pages. `requirements-ocr.txt` points pip to PaddlePaddle's CUDA 12.6 index.

   ```powershell
   py -3.12 -m venv .venv-ocr
   .venv-ocr\Scripts\python -m pip install -r requirements-ocr.txt
   ```

3. Install Ollama and download the language model (5.2 GB).

   ```powershell
   ollama pull qwen3:8b
   ```

4. Download the documents (about 270 MB of PDFs from the UCSF Industry Documents Library) and render their first pages. This takes 15 to 30 minutes; if it stops, run it again and it continues where it left off.

   ```powershell
   .venv\Scripts\python -m pipeline.01_sample.download_pdfs
   ```

5. Build the app's archive.

   ```powershell
   .venv\Scripts\python -m app.build_archive
   ```

   It prints `Archive: 766 filed records` and `Review queue: 480 pages`; the app shows 761 records, because five pages are withheld for personal data.

The embedding model (Qwen3-Embedding-0.6B, 1.2 GB) and the PaddleOCR models download on first use.

## Run the app

```powershell
.venv\Scripts\python -m streamlit run app\app.py
```

The app opens at http://localhost:8501. Search works on its own; Ask, and the record-type suggestion for new uploads, need Ollama running. The first upload takes about 40 seconds while the OCR and embedding models load.

The name stamped on uploads, approvals and corrections is "Local user". To use another name, set `LOCALRIM_USER` before starting the app:

```powershell
$env:LOCALRIM_USER = "Your Name"
```

## Re-run the pipeline

Each step folder has a README with its commands, inputs and outputs. Run every command from the repository root: the scripts run as modules (`python -m pipeline.<step>.<script>`), and `pipeline/common.py` holds the paths and settings they share. All saved results are included, so any step can be run on its own. The steps that need no OCR or language model reproduce the saved results exactly; the language-model steps run at temperature 0 with a fixed seed.

```powershell
.venv\Scripts\python -m pipeline.03_classify.train_tfidf
.venv\Scripts\python -m pipeline.06_site_data.build_site_data
```

## Data

The documents belong to the [UCSF Industry Documents Library](https://www.industrydocuments.ucsf.edu/); see its [Copyright and Fair Use](https://www.industrydocuments.ucsf.edu/about-idl/copyright/) page. The PDFs and page images are not included, apart from the seven example pages in [examples/](examples/). What is included, and how to request removal, is described in [data/README.md](data/README.md).

## Licence

MIT, see [LICENSE](LICENSE). Third-party software, models, fonts and data are listed in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
