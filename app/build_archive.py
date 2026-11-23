"""Build the app's archive and review queue from the saved results.

Reads the labels, the scale-run predictions and suggestions, the manifest and the OCR
results, and writes new files under outputs/app/:

    archive.csv      one row per filed record
    queue.csv        one row per page waiting for review
    thumbs/<id>.jpg  page thumbnails (made once, skipped if present)

The archive holds filed records only:
- labeled pages (training and final test, excluded pages left out) are filed under their
  labels, with the "also fits" choice as a cross-reference;
- scale-run pages the pipeline filed itself are filed under the classifier's department.
  The record type is the language model's suggestion, marked "suggested", and is kept
  only when its department agrees with the classifier's (Administration counted with
  Legal); otherwise the record has no record type yet.
Pages still waiting for a person go to the review queue.

Per record: the document date in ISO form and as written (partial dates stay partial,
"undated" when there is none), the file name (title_ID_date), page count, reading quality
from the PaddleOCR run (weak = fewer than 20 words) and how it was filed.

Usage:
    python -m app.build_archive
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd
import yaml
from PIL import Image

from pipeline.common import MERGE_FOR_SCORING as MERGE, ROOT, TEST_LABELS, TRAIN_LABELS

DATA = ROOT / "data" / "pm_sample"
OUT = ROOT / "outputs" / "app"
THUMBS = OUT / "thumbs"
TAXONOMY = ROOT / "taxonomy" / "rim_taxonomy_final.yaml"
ID_PREFIX = "PM"          # set per client
WEAK_WORDS = 20           # reading too weak to search well (same idea as the 20-token route rule)
THUMB_WIDTH = 260
MONTHS = {m: i for i, m in enumerate(["january", "february", "march", "april", "may", "june", "july",
                                      "august", "september", "october", "november", "december"], 1)}


def latest(path: Path) -> pd.DataFrame:
    """Read a labels file, keeping each document's latest answer."""
    df = pd.read_csv(path, dtype=str).fillna("")
    return df.sort_values("saved_at").drop_duplicates("doc_id", keep="last")


def jsonl(path: Path) -> dict:
    """Read a JSON-lines file into id -> record."""
    out = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            out[r["id"]] = r
    return out


def doc_date(raw: str) -> tuple[str, str]:
    """Convert a catalogue date to (ISO, as shown).

    '1982 January 01' -> ('1982-01-01', 'January 1, 1982'); '1987 July' -> ('1987-07', 'July 1987');
    '1988' -> ('1988', '1988'); '' -> ('undated', 'Undated').
    """
    parts = raw.strip().split()
    if not parts or not re.fullmatch(r"\d{4}", parts[0]):
        return "undated", "Undated"
    year = parts[0]
    if len(parts) >= 2 and parts[1].lower() in MONTHS:
        month = MONTHS[parts[1].lower()]
        name = parts[1].capitalize()
        if len(parts) >= 3 and parts[2].isdigit():
            day = int(parts[2])
            return f"{year}-{month:02d}-{day:02d}", f"{name} {day}, {year}"
        return f"{year}-{month:02d}", f"{name} {year}"
    return year, year


def slug(text: str, limit: int = 50) -> str:
    """Return text as a lowercase, hyphenated slug of at most limit characters."""
    s = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return s[:limit].rstrip("-") or "untitled"


def file_id(bates: str, doc_id: str) -> str:
    """Return the file ID: the prefix plus the first Bates number, or the document ID."""
    first = bates.split("-")[0].strip()
    return f"{ID_PREFIX}{first}" if first else doc_id


def make_thumb(doc_id: str) -> None:
    """Make a page thumbnail unless it exists or the page image is missing."""
    target = THUMBS / f"{doc_id}.jpg"
    src = DATA / "images_native" / f"{doc_id}.png"
    if target.exists() or not src.exists():
        return
    with Image.open(src) as im:
        im = im.convert("RGB")
        h = int(im.height * THUMB_WIDTH / im.width)
        im.resize((THUMB_WIDTH, h)).save(target, quality=80)


def main() -> None:
    """Build archive.csv, queue.csv and the thumbnails."""
    OUT.mkdir(parents=True, exist_ok=True)
    THUMBS.mkdir(parents=True, exist_ok=True)
    tax = yaml.safe_load(TAXONOMY.read_text(encoding="utf-8"))
    dept_name = {f["id"]: f["name"] for f in tax["functions"]}
    series_of, series_name, type_name = {}, {}, {}
    for f in tax["functions"]:
        for s in f["series"]:
            series_name[s["id"]] = s["name"]
            for t in s["types"]:
                series_of[t["id"]], type_name[t["id"]] = s["id"], t["name"]

    def path_of(type_id: str) -> str:
        if not type_id:
            return ""
        return f"{dept_name.get(type_id.split('.')[0], '')} › {series_name.get(series_of.get(type_id, ''), '')} › {type_name.get(type_id, '')}"

    manifest = pd.read_csv(DATA / "manifest.csv", dtype=str).fillna("").set_index("id")
    pdf_info = pd.read_csv(DATA / "pdf_info.csv", dtype={"id": str}).set_index("id")
    ocr = pd.read_csv(DATA / "ocr_paddle" / "ocr_results.csv", dtype={"id": str}, keep_default_na=False).set_index("id")

    def base(doc_id: str) -> dict:
        m = manifest.loc[doc_id]
        iso, shown = doc_date(m["documentdate"])
        words = int(ocr.loc[doc_id, "words"]) if doc_id in ocr.index else 0
        conf = float(ocr.loc[doc_id, "mean_conf"]) if doc_id in ocr.index else 0.0
        return {"id": doc_id, "file_id": file_id(m["bates"], doc_id), "bates": m["bates"],
                "catalogue_title": m["title"], "date_iso": iso, "date_shown": shown,
                "decade": f"{iso[:3]}0s" if iso != "undated" else "Undated",
                "pages": int(pdf_info.loc[doc_id, "pdf_pages"]) if doc_id in pdf_info.index else 1,
                "ocr_words": words, "ocr_conf": conf, "weak_reading": words < WEAK_WORDS,
                "image": f"data/pm_sample/images_native/{doc_id}.png", "pdf": f"data/pm_sample/pdf/{doc_id}.pdf",
                "text": f"data/pm_sample/ocr_paddle/{doc_id}.txt", "thumb": f"outputs/app/thumbs/{doc_id}.jpg"}

    def finish(r: dict) -> dict:
        title = r["catalogue_title"].strip() or (type_name.get(r["record_type"], "") or "Untitled")
        r["title"] = title[:1] + title[1:].lower() if title.isupper() else title
        r["file_name"] = f"{slug(r['title'])}_{r['file_id']}_{r['date_iso']}.pdf"
        return r

    def record(doc_id, department, type_id, suggested, also_fits, how, added, reason=""):
        r = base(doc_id)
        s = series_of.get(type_id, "")
        r.update({"department": department, "department_name": dept_name.get(department, department),
                  "series": s, "series_name": series_name.get(s, ""),
                  "record_type": type_id, "record_type_name": type_name.get(type_id, ""),
                  "suggested": suggested, "reason": reason,
                  "also_fits": also_fits, "also_fits_path": path_of(also_fits),
                  "how_filed": how, "added": added})
        return finish(r)

    rows, labeled = [], set()
    for path, phase in [(TRAIN_LABELS, "labeling"), (TEST_LABELS, "final test labeling")]:
        for r in latest(path).itertuples():
            labeled.add(r.doc_id)
            if r.outcome == "labeled" and r.function_id != "HR":
                rows.append(record(r.doc_id, r.function_id, r.type_id, False, r.also_fits,
                                   f"Filed by Salo during {phase}", r.saved_at[:10]))

    preds = pd.read_csv(ROOT / "results" / "scale_run" / "predictions.csv", dtype={"id": str})
    sug = {**jsonl(ROOT / "results" / "scale_run" / "suggestions_A.jsonl"),
           **jsonl(ROOT / "results" / "scale_run" / "suggestions_A_autofiled.jsonl")}
    queue = []
    for p in preds.itertuples():
        if p.id in labeled:
            continue
        s = sug.get(p.id, {})
        t = s.get("record_type", "")
        t_dept = t.split(".")[0] if t else ""
        if p.routed == "auto-filed":
            agrees = MERGE.get(t_dept, t_dept) == p.predicted
            dept = t_dept if agrees else p.predicted
            rows.append(record(p.id, dept, t if agrees else "", agrees, "",
                               "Filed by the pipeline in the scale run (counts as approved, App-16)", "2026-11-12",
                               s.get("reason", "") if agrees else ""))
        else:
            q = base(p.id)
            q.update({"route_reason": p.route_reason, "classifier_department": p.predicted,
                      "confidence": p.confidence, "second": p.second, "second_confidence": p.second_confidence,
                      "suggested_department": t_dept, "suggested_record_type": t,
                      "suggested_path": path_of(t), "second_record_type": s.get("second_record_type", ""),
                      "reason": s.get("reason", ""), "sources": "; ".join(s.get("sources", [])),
                      "record_type": t})
            queue.append(finish(q))

    archive = pd.DataFrame(rows).sort_values(["department", "series", "record_type", "date_iso"])
    archive.to_csv(OUT / "archive.csv", index=False)
    pd.DataFrame(queue).to_csv(OUT / "queue.csv", index=False)

    print("Making thumbnails (first run only)...", flush=True)
    for doc_id in archive["id"]:
        make_thumb(doc_id)

    auto = archive["how_filed"].str.startswith("Filed by the pipeline")
    print(f"Archive: {len(archive)} filed records ({(~auto).sum()} by Salo, {auto.sum()} by the pipeline; "
          f"{(archive['record_type'] == '').sum()} without a record type yet; "
          f"{archive['weak_reading'].sum()} with a weak reading; {(archive['date_iso'] == 'undated').sum()} undated)")
    print(f"Review queue: {len(queue)} pages")
    print(f"Saved to {OUT}")


if __name__ == "__main__":
    main()
