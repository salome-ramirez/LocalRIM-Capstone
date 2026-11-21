# 06 Site data

Collects every number and chart the project site shows into one file, `docs/data/site_data.json`.

Site chapters: all of them.

| Script | What it does |
|---|---|
| `build_site_data.py` | Reads the labels, taxonomy and result tables and writes `docs/data/site_data.json` |
| `localrim_metrics.py` | The labeling counts, defined once for the site and the batch records |
| `batch_metrics.py` | Per-batch labeling counts, `results/batch_metrics.csv`; the build stops if they disagree with the site |

Command, from the repository root (after `app\build_archive.py`, which builds the app's archive the site counts):

```powershell
.venv\Scripts\python app\build_archive.py
.venv\Scripts\python pipeline\06_site_data\build_site_data.py
```

It writes counts, scores and rates only. Run on this repository, it reproduces the published `site_data.json` exactly; only the `updated` time changes.
