"""Settings and helpers shared by the pipeline steps and the app.

Run the pipeline scripts from the repository root as modules, for example
    python -m pipeline.03_classify.train_tfidf
Importing this module also puts every step folder and app/ on the import path, so a
script can import another step's module by name (from ocr_eda import ...) and the app's
package (from localrim import ...).
"""
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

STEPS = ("01_sample", "02_ocr", "03_classify", "04_route", "05_validate", "06_site_data")
sys.path[:0] = [p for p in [str(ROOT / "pipeline" / s) for s in STEPS] + [str(ROOT / "app")] if p not in sys.path]

# Label files, as the labeling apps save them (labels/<set>_<annotator>.csv)
TRAIN_LABELS = ROOT / "labels" / "label_pm_labels_salo.csv"
TEST_LABELS = ROOT / "labels" / "final_test_labels_salo.csv"
SPOT_LABELS = ROOT / "labels" / "spotcheck_salo.csv"

MERGE_FOR_SCORING = {"ADM": "LEG"}   # Administration is scored with Legal


def done_ids(path: Path) -> set[str]:
    """Return the IDs already saved in a JSON-lines output, so a rerun continues where it stopped."""
    if not path.exists():
        return set()
    return {json.loads(line)["id"] for line in path.read_text(encoding="utf-8").splitlines() if line.strip()}


def resume_csv(path: Path, key: str = "id", dtype=str) -> set[str]:
    """Drop the rows that failed from a results CSV (they are tried again) and return the keys done."""
    if not path.exists():
        return set()
    import pandas as pd
    res = pd.read_csv(path, dtype=dtype, keep_default_na=False)
    res = res[res["error"] == ""]
    res.to_csv(path, index=False)
    return set(res[key])


class Progress:
    """Print "n/total pages | x s per page | about m min left" every `every` items and after the last."""

    def __init__(self, total: int, every: int = 10, label: str = "", unit: str = "page"):
        self.total, self.every, self.label, self.unit = total, every, label, unit
        self.start = time.time()

    def __call__(self, n: int, extra: str = "") -> None:
        if n % self.every == 0 or n == self.total:
            per = (time.time() - self.start) / n
            print(f"{self.label}{n}/{self.total} {self.unit}s | {per:.1f} s per {self.unit} | "
                  f"about {per * (self.total - n) / 60:.0f} min left{extra}", flush=True)
