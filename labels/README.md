# Labels

Hand labels made with `pipeline/01_sample/labeler.py` and the spot-check app. Each row is one save; the newest save per document counts.

| File | Contents |
|---|---|
| `label_pm_labels_salo.csv` | the 262 reviewed training documents (250 valid labels) |
| `final_test_labels_salo.csv` | the blind final-test labels (150 valid) |
| `skipped_pm_salo.csv` | documents set aside during labeling (failed downloads) |
| `spotcheck_salo.csv` | the blind spot-check of 20 auto-filed scale-run pages |
| `skim_labels.csv`, `skim_v2_labels.csv` | the two pilot labeling rounds behind the taxonomy (generic categories, then Philip Morris departments) |

The file names carry the annotator's name because the scripts find label files by that pattern.

Free-text notes are not published. The `note` column keeps only what the counts read: the exclusion reason of excluded pages (for example "blank or unreadable") and the marker "Non-English".
