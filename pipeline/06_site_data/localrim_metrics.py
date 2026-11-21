"""Labeling metrics shared by the site data and the batch records, so they always agree.

Definitions:
- Labels: every annotator's file, newest save per document.
- Reviewed: documents with a saved answer (labeled, excluded or no fit).
- Valid: outcome "labeled".
- Unsure share: unsure / reviewed.
- Skipped: in a skip file and not labeled since.
- Batch: position in the manifest (labeling order), 100 per batch.
- Data as of: time of the newest saved label.
"""
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
BATCH_SIZE = 100


def _read(path):
    """Read a CSV, falling back to cp1252 for files saved by Excel."""
    try:
        return pd.read_csv(path, dtype=str, keep_default_na=False)
    except UnicodeDecodeError:   # files saved by Excel on Windows
        return pd.read_csv(path, dtype=str, keep_default_na=False, encoding="cp1252")


# Optional cutoff (ISO time): count only what had been saved by then. Used to
# rebuild snapshots of earlier rounds. None = everything (normal use).
AS_OF = None


def all_saves():
    """Return every save, re-saves included."""
    frames = [_read(f) for f in sorted((ROOT / "labels").glob("label_pm_labels_*.csv"))]
    saves = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(
        columns=["path", "outcome", "function_id", "type_id", "also_fits", "certainty", "note", "saved_at"])
    return saves[saves["saved_at"] <= AS_OF] if AS_OF else saves


def load_labels():
    """Return the newest answer per document."""
    return all_saves().sort_values("saved_at").drop_duplicates("path", keep="last").reset_index(drop=True)


def load_manifest():
    """Return the sample manifest."""
    return _read(ROOT / "data" / "pm_sample" / "manifest.csv")


def load_skips(labels):
    """Return the skipped documents not labeled since."""
    frames = [_read(f) for f in sorted((ROOT / "labels").glob("skipped_pm_*.csv"))]
    skips = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=["path", "saved_at"])
    if AS_OF:
        skips = skips[skips["saved_at"] <= AS_OF]
    return skips[~skips["path"].isin(labels["path"])].drop_duplicates("path")


def summary(labels):
    """Return the headline numbers."""
    reviewed = len(labels)
    unsure = int((labels["certainty"] == "unsure").sum())
    return {
        "reviewed": reviewed,
        "valid": int((labels["outcome"] == "labeled").sum()),
        "excluded": int((labels["outcome"] == "exclude").sum()),
        "no_fit": int((labels["outcome"] == "no_fit").sum()),
        "unsure": unsure,
        "unsure_share": round(unsure / reviewed, 3) if reviewed else None,
        "data_as_of": labels["saved_at"].max() if reviewed else None,
    }


def batch_of(paths, manifest):
    """Return each path's batch number (manifest position, 100 per batch)."""
    position = {p: i + 1 for i, p in enumerate(manifest["image"])}
    return (paths.map(position) - 1) // BATCH_SIZE + 1
