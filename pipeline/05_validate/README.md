# 05 Validate

Freezes the pipeline and tests it once on 150 pages it has never seen, then runs it on 1,000 new documents and checks the app's archive assistant.

Site chapters: **Classification, Routing and Validation** (final test, scale run), **Findings** and **The app**.

| Script | What it does | Reads | Writes |
|---|---|---|---|
| `final_test.py` | The one-time final test: predict without reading the labels, then score once | embeddings, folds, labels, cleaned text, taxonomy | `results/final_test/` |
| `scale_run.py` | The frozen pipeline on the 1,000-document scale batch: how much is filed alone, why the rest goes to a person, how long it takes | as above, OCR timings | `results/scale_run/` |
| `spotcheck.py` | Blind spot-check app for 20 auto-filed scale-run pages | spot-check sample, page images | `labels/spotcheck_<annotator>.csv` |
| `score_record_types.py` | Scores the record-type suggestions at every taxonomy level | suggestions, labels, taxonomy | `results/record_types/` |
| `archive_check.py` | Does the app's assistant find, cite and decline correctly? | `outputs/app/archive.csv` | `results/archive_check/` |

Commands, from the repository root (Ollama running with `qwen3:8b`):

```powershell
.venv\Scripts\python -m pipeline.05_validate.final_test --stage predict
.venv\Scripts\python -m pipeline.05_validate.final_test --stage score
.venv\Scripts\python -m pipeline.05_validate.scale_run
.venv\Scripts\python -m pipeline.05_validate.score_record_types
.venv\Scripts\python -m app.build_archive
.venv\Scripts\python -m pipeline.05_validate.archive_check
```

`final_test.py --stage score` refuses to run when the scores exist (the test is scored once); `--force` only rebuilds the same files.

Headline results of the final test: macro F1 0.60 (TF-IDF also 0.60), 41% of pages filed automatically, 54 of those 62 right (87%, against a 90% target).

The "faithful" column of the archive check was filled in by a person after reading each answer against its cited pages; a re-run leaves it blank.
