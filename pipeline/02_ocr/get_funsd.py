"""Download the FUNSD forms dataset (199 scanned forms with word-level ground truth).

FUNSD is used to measure each OCR engine against known text. It is not included in this
repository; this script downloads it from the Hugging Face Hub (https://guillaumejaume.github.io/FUNSD/).

Writes:
    data/funsd/<split>/<doc_id>.png    the scanned form
    data/funsd/<split>/<doc_id>.json   words, boxes and tags for the form
    data/funsd/manifest.csv

Usage:
    python -m pipeline.02_ocr.get_funsd
"""

import csv
import json
from datasets import load_dataset

from pipeline.common import ROOT

DATASET_ID = "nielsr/funsd"
OUT_DIR = ROOT / "data" / "funsd"


def main():
    """Download both splits and save each form's image, fields and the manifest."""
    rows = []
    for split in ["train", "test"]:
        ds = load_dataset(DATASET_ID, split=split)   # small, so a normal download is fine
        split_dir = OUT_DIR / split
        split_dir.mkdir(parents=True, exist_ok=True)

        image_col = next(name for name, feat in ds.features.items()
                         if type(feat).__name__ == "Image")

        # Tag columns are stored as numbers; keep their names so the JSON is readable
        tag_names = {name: feat.feature.names for name, feat in ds.features.items()
                     if hasattr(getattr(feat, "feature", None), "names")}

        for i, example in enumerate(ds):
            doc_id = f"funsd_{split}_{i:03d}"
            file_path = split_dir / f"{doc_id}.png"
            ann_path = split_dir / f"{doc_id}.json"
            example[image_col].save(file_path)

            # Save every field except the image itself (words, boxes, tags, original id)
            annotation = {k: v for k, v in example.items() if k != image_col}
            annotation["tag_names"] = tag_names
            with open(ann_path, "w", encoding="utf-8") as f:
                json.dump(annotation, f, ensure_ascii=False)

            rows.append({"doc_id": doc_id, "file_path": file_path.as_posix(),
                         "annotation_path": ann_path.as_posix(),
                         "source_dataset": DATASET_ID, "source_split": split,
                         "source_index": i})
        print(f"{split}: saved {len(ds)} images and annotations")
        print(f"  fields saved: {[k for k in ds.features if k != image_col]}")

    with open(OUT_DIR / "manifest.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {len(rows)} FUNSD images to {OUT_DIR}")


if __name__ == "__main__":
    main()
