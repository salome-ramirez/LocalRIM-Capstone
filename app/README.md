# App

The LocalRIM application (Streamlit): the Archive (search, ask, browse, read), Add files, Intake and the Review queue.

| File | What it is |
|---|---|
| `app.py` | the entry point: page setup, data loading, the masthead and the screen switch |
| `screens/records.py` | Archive home, Browse and the Reader |
| `screens/assistant.py` | the question box, its answer and sources |
| `screens/intake.py` | Add files and the Intake screen |
| `screens/review.py` | the Review queue |
| `screens/shared.py` | data loading, history, search, related records and saved searches shared by the screens |
| `build_archive.py` | builds the Archive and the review queue from the saved results (`outputs/app/archive.csv`, `queue.csv`) |
| `localrim/assistant.py` | search and question answering |
| `localrim/intake.py` | the records behind Add files, Intake and the Review queue |
| `localrim/pipeline.py` | one page in, one filing decision out (OCR, classifier, routing, record-type suggestion) |
| `localrim/ocr_worker.py` | the PaddleOCR helper, run in `.venv-ocr` |
| `static/fonts/` | self-hosted fonts |

The pipeline uses the frozen settings of the research steps, imported from `pipeline/` (cutoff 0.55, 20-token rule, classifier, prompt).

```powershell
.venv\Scripts\python -m app.build_archive
.venv\Scripts\python -m streamlit run app\app.py
```

Files the app writes (uploads, approvals, corrections, saved searches) go to `outputs/app/`, which is not tracked.
