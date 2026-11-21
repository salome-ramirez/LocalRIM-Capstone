"""Records behind Add files, Intake and the Review queue.

Nothing enters the Archive without a person:
- Add files: each upload becomes a batch. Every page is read and filed by the intake
  pipeline (localrim.pipeline.Pipeline), and the language model suggests a title and the
  document date. A page is "Ready to approve" when the pipeline would file it on its own
  (confidence at or above the 0.55 cutoff and at least 20 tokens of text) and its file
  number isn't already filed. Otherwise it "Needs review", with the reason: confidence
  below the cutoff, weak reading, or possible duplicate.
- Intake: approving files the selected ready pages in the Archive; flagged pages go to
  the Review queue. Approval confirms the department; record types stay "suggested"
  unless edited. An approval can be undone for 10 minutes.
- Review queue: the scale-run pages waiting for a person, plus flagged uploads. "File to
  Archive" files a page with the reviewer's choices.

Files in outputs/app/, all append-only logs or rebuilt views (archive.csv and queue.csv
from app/build_archive.py are never changed):
  batches.csv        one row per batch: uploaded and approved by whom and when, counts
  intake.csv         one row per uploaded page, with its status (waiting, approved,
                     to review, filed)
  filed.csv          records filed in the app, same columns as archive.csv; the app
                     shows archive.csv + filed.csv
  review_log.csv     Review queue actions: filed, skipped, second look requested
  new_vectors.npy / new_vector_ids.csv   page embeddings of uploads, for search
  new_neighbors.csv  the five most similar filed pages of each upload (S1-S5), same
                     columns as results/scale_run/neighbors.csv
  uploads/<batch>/, intake_text/, thumbs/   stored pages, their OCR text and thumbnails
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
APP = ROOT / "outputs" / "app"
UPLOADS = APP / "uploads"
TEXTS = APP / "intake_text"
THUMBS = APP / "thumbs"
BATCHES = APP / "batches.csv"
INTAKE = APP / "intake.csv"
FILED = APP / "filed.csv"
REVIEW_LOG = APP / "review_log.csv"
NEW_VECS = APP / "new_vectors.npy"
NEW_VEC_IDS = APP / "new_vector_ids.csv"
OLLAMA = "http://localhost:11434/api/chat"
LLM = "qwen3:8b"
ID_PREFIX = "PM"      # set per client
MERGE = {"ADM": "LEG"}
WEAK_WORDS = 20
UNDO_MINUTES = 10
MONTH_NAMES = ["January", "February", "March", "April", "May", "June", "July", "August",
               "September", "October", "November", "December"]

ARCHIVE_COLS = ["id", "file_id", "bates", "catalogue_title", "date_iso", "date_shown", "decade", "pages",
                "ocr_words", "ocr_conf", "weak_reading", "image", "pdf", "text", "thumb", "department",
                "department_name", "series", "series_name", "record_type", "record_type_name", "suggested",
                "reason", "also_fits", "also_fits_path", "how_filed", "added", "title", "file_name"]
# Pages flagged for personal data during labeling: hidden from every screen, nothing deleted.
HIDDEN = {"smmw0220", "nkjv0160", "tqyl0166", "tzmx0220", "kpvf0220"}
REASONS = {"below cutoff": "Confidence below the 55% cutoff", "under 20 tokens": "Weak reading",
           "no readable text": "Weak reading", "duplicate": "Possible duplicate"}


def now() -> str:
    """Return the current time as 'YYYY-MM-DD HH:MM'."""
    return datetime.now().strftime("%Y-%m-%d %H:%M")


def read(path: Path) -> pd.DataFrame:
    """Read a CSV log as strings; an empty frame if the file doesn't exist."""
    return pd.read_csv(path, dtype=str).fillna("") if path.exists() else pd.DataFrame()


def append(path: Path, rows: list[dict]) -> None:
    """Append rows to a CSV log."""
    old = read(path)
    pd.concat([old, pd.DataFrame(rows)], ignore_index=True).to_csv(path, index=False)


# Names and dates
def slug(text: str, limit: int = 50) -> str:
    """Return text as a lowercase, hyphenated slug of at most limit characters."""
    s = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return s[:limit].rstrip("-") or "untitled"


def date_shown(iso: str) -> str:
    """Format an ISO date (full or partial) for display."""
    if not iso or iso == "undated":
        return "Undated"
    p = iso.split("-")
    if len(p) == 3:
        return f"{MONTH_NAMES[int(p[1]) - 1]} {int(p[2])}, {p[0]}"
    if len(p) == 2:
        return f"{MONTH_NAMES[int(p[1]) - 1]} {p[0]}"
    return p[0]


def clean_date(raw: str) -> str:
    """Keep only a valid ISO date, partial allowed; anything else is 'undated'."""
    raw = (raw or "").strip()
    for pat in (r"(1[89]\d\d|20[0-2]\d)-(0[1-9]|1[0-2])-(0[1-9]|[12]\d|3[01])", r"(1[89]\d\d|20[0-2]\d)-(0[1-9]|1[0-2])",
                r"(1[89]\d\d|20[0-2]\d)"):
        if re.fullmatch(pat, raw):
            return raw
    return "undated"


def file_name(title: str, file_id: str, iso: str) -> str:
    """Return the record's file name: title_ID_date.pdf."""
    return f"{slug(title)}_{file_id}_{iso or 'undated'}.pdf"


def suggest_title_date(text: str) -> tuple[str, str]:
    """Ask the language model for a new page's title and document date.

    Returns ('', 'undated') when Ollama can't be reached. A person confirms both at approval.
    """
    import requests
    flat = " ".join(text.split())[:2500]
    if not flat:
        return "", "undated"
    prompt = ("Below is the OCR text of the first page of a scanned company document.\n"
              "Give (1) a short title for the document, 3 to 10 words, in sentence case, as an archivist "
              "would catalogue it (e.g. 'Outdoor bulletin display contract'), and (2) the document's own "
              "date if the page shows one, as YYYY-MM-DD, YYYY-MM or YYYY; empty if no date is shown. "
              "Do not guess a date that is not on the page.\n"
              'Answer in JSON: {"title": "...", "date": "..."}\n\nPAGE:\n' + flat)
    try:
        r = requests.post(OLLAMA, timeout=120, json={
            "model": LLM, "stream": False, "think": False, "format": "json",
            "options": {"temperature": 0, "seed": 44, "num_ctx": 4096},
            "messages": [{"role": "user", "content": prompt}]})
        r.raise_for_status()
        ans = json.loads(r.json()["message"]["content"])
        title = " ".join(str(ans.get("title", "")).split())[:120]
        return title, clean_date(str(ans.get("date", "")))
    except Exception:
        return "", "undated"


# Add files: one batch
def next_batch() -> int:
    """Return the next batch number."""
    b = read(BATCHES)
    return int(b["batch"].astype(int).max()) + 1 if len(b) else 1


def known_file_ids(archive: pd.DataFrame, queue: pd.DataFrame) -> set[str]:
    """Return the file IDs already filed, queued or waiting in Intake."""
    ids = set(archive["file_id"]) | set(queue.get("file_id", pd.Series(dtype=str)))
    i = read(INTAKE)
    if len(i):
        ids |= set(i.loc[i["status"].isin(["waiting", "to review", "approved", "filed"]), "file_id"])
    return {x for x in ids if x}


def save_upload(batch: int, k: int, name: str, data: bytes) -> tuple[str, Path, Path]:
    """Store an upload as <id>.<ext>, with a PDF copy for PNGs so every record downloads as PDF."""
    page_id = f"b{batch:03d}-{k:02d}"
    folder = UPLOADS / f"batch_{batch:03d}"
    folder.mkdir(parents=True, exist_ok=True)
    ext = Path(name).suffix.lower()
    src = folder / f"{page_id}{ext}"
    src.write_bytes(data)
    pdf = src
    if ext != ".pdf":
        pdf = folder / f"{page_id}.pdf"
        with Image.open(src) as im:
            im.convert("RGB").save(pdf, "PDF", resolution=300)
    return page_id, src, pdf


def pdf_pages(pdf: Path) -> int:
    """Return a PDF's page count (1 if it can't be read)."""
    try:
        import fitz
        with fitz.open(pdf) as d:
            return d.page_count
    except Exception:
        return 1


def make_thumb(src: Path, page_id: str, width: int = 260) -> Path:
    """Save a thumbnail of a page image and return its path."""
    THUMBS.mkdir(parents=True, exist_ok=True)
    target = THUMBS / f"{page_id}.jpg"
    with Image.open(src) as im:
        im = im.convert("RGB")
        im.resize((width, int(im.height * width / im.width))).save(target, quality=80)
    return target


def rel(p: Path) -> str:
    """Return a path relative to the repository root, with forward slashes."""
    return Path(p).resolve().relative_to(ROOT).as_posix()


def process_page(pipe, batch: int, k: int, name: str, data: bytes, known: set[str], type_info: dict,
                 dept_name: dict) -> tuple[dict, np.ndarray | None]:
    """Read and file one upload with the pipeline; return the intake row and its vector."""
    page_id, src, pdf = save_upload(batch, k, name, data)
    r = pipe.file(src)
    base = {"id": page_id, "batch": str(batch), "original_name": name, "status": "waiting", "pdf": rel(pdf),
            "pages": str(pdf_pages(pdf)), "added": now()}
    if r.get("error"):
        return {**base, "status": "to review", "route": "Needs review", "route_reason": "Weak reading",
                "error": r["error"], "image": rel(src) if src.suffix != ".pdf" else ""}, None
    image = Path(r["image"])
    TEXTS.mkdir(parents=True, exist_ok=True)
    (TEXTS / f"{page_id}.txt").write_text(r["text"], encoding="utf-8")
    fid = f"{ID_PREFIX}{r['file_number']}" if r["file_number"] else page_id
    duplicate = bool(r["file_number"]) and fid in known

    sug = r.get("suggestion", {})
    t = sug.get("record_type", "") if not sug.get("error") else ""
    t_dept = t.split(".")[0] if t else ""
    agrees = bool(t) and MERGE.get(t_dept, t_dept) == r["department"]
    dept = t_dept if agrees else r["department"]
    record_type = t if agrees else ""
    d, dn, s, sn, tn = type_info.get(record_type, (dept, dept_name.get(dept, dept), "", "", ""))

    title, iso = suggest_title_date(r["text"])
    title = title or (tn or "Untitled")
    if r["routed"] == "auto-filed" and not duplicate:
        route, why = "Ready to approve", ""
    else:
        route, why = "Needs review", REASONS["duplicate"] if duplicate else REASONS.get(r["route_reason"], r["route_reason"])
    words = int(r["words"])
    row = {**base, "file_id": fid, "bates": r["file_number"], "image": rel(image), "text": rel(TEXTS / f"{page_id}.txt"),
           "thumb": rel(make_thumb(image, page_id)), "ocr_words": str(words), "ocr_conf": f"{float(r['ocr_confidence']):.1f}",
           "weak_reading": str(words < WEAK_WORDS),
           "classifier_department": r["department"], "confidence": f"{r['confidence']:.3f}",
           "second": r["second"], "second_confidence": f"{r['second_confidence']:.3f}",
           "department": dept, "department_name": dept_name.get(dept, dept), "series": s, "series_name": sn,
           "record_type": record_type, "record_type_name": tn,
           "suggested_record_type": t, "second_record_type": sug.get("second_record_type", ""),
           "reason": sug.get("reason", ""), "sources": "; ".join(sug.get("sources", [])),
           "title": title, "date_iso": iso, "date_shown": date_shown(iso),
           "file_name": file_name(title, fid, iso), "route": route, "route_reason": why,
           "duplicate_of": fid if duplicate else "", "error": sug.get("error", "")}
    row["_neighbors"] = r.get("neighbors", [])   # saved to new_neighbors.csv by save_batch, not to intake.csv
    if route == "Needs review":
        row["status"] = "to review"
    known.add(fid)
    vec = np.asarray(r["vector"], dtype=np.float32) if "vector" in r else None
    return row, vec


def save_batch(batch: int, rows: list[dict], vecs: list, user: str) -> None:
    # Similar filed pages in S1-S5 order, so "filed page 1" in a reason opens the right page.
    """Save a processed batch: intake rows, similar pages, vectors and the batch record."""
    nbs = [{"id": r["id"], "rank": str(k), "neighbor_id": nb["id"], "neighbor_department": nb["department"],
            "similarity": str(nb["similarity"])}
           for r in rows for k, nb in enumerate(r.pop("_neighbors", []) or [], 1)]
    if nbs:
        append(Path(NEW_VEC_IDS).parent / "new_neighbors.csv", nbs)
    append(INTAKE, rows)
    good = [(r["id"], v) for r, v in zip(rows, vecs) if v is not None]
    if good:
        ids = read(NEW_VEC_IDS)["id"].tolist() if NEW_VEC_IDS.exists() else []
        old = np.load(NEW_VECS) if NEW_VECS.exists() else np.zeros((0, len(good[0][1])), np.float32)
        np.save(NEW_VECS, np.vstack([old, np.stack([v for _, v in good])]))
        pd.DataFrame({"id": ids + [i for i, _ in good]}).to_csv(NEW_VEC_IDS, index=False)
    ready = sum(r["route"] == "Ready to approve" for r in rows)
    append(BATCHES, [{"batch": str(batch), "uploaded_by": user, "uploaded_at": now(), "pages": str(len(rows)),
                      "ready": str(ready), "to_review": str(len(rows) - ready),
                      "approved_by": "", "approved_at": "", "approved": ""}])


# Intake: approve and undo
def to_record(row: dict, how: str, suggested: bool) -> dict:
    """Turn an intake or queue row into an archive record."""
    out = {c: row.get(c, "") for c in ARCHIVE_COLS}
    iso = out["date_iso"] or "undated"
    out.update({"catalogue_title": "", "decade": f"{iso[:3]}0s" if iso != "undated" else "Undated",
                "suggested": str(suggested and bool(out["record_type"])), "also_fits": "", "also_fits_path": "",
                "how_filed": how, "added": now()[:10],
                "file_name": file_name(out["title"], out["file_id"], iso)})
    return out


def apply_edit(row: dict, title: str, iso: str, type_id: str, type_info: dict) -> tuple[dict, bool]:
    """Apply what the person changed in Intake; return the row and whether the record type changed."""
    row = dict(row)
    row["title"] = title.strip() or row["title"]
    row["date_iso"] = clean_date(iso)
    row["date_shown"] = date_shown(row["date_iso"])
    edited = type_id != row.get("record_type", "")
    if edited and type_id in type_info:
        d, dn, s, sn, tn = type_info[type_id]
        row.update({"department": d, "department_name": dn, "series": s, "series_name": sn,
                    "record_type": type_id, "record_type_name": tn})
    return row, edited


def approve(batch: str, edits: dict, user: str, type_info: dict) -> int:
    """File the edited pages (page id -> (title, date, record type)) and log the approval."""
    intake = read(INTAKE)
    filed = []
    for pid, (title, iso, type_id) in edits.items():
        m = intake["id"] == pid
        row, edited = apply_edit(intake[m].iloc[0].to_dict(), title, iso, type_id, type_info)
        filed.append(to_record(row, f"Approved in Intake by {user} (batch {batch})", suggested=not edited))
        intake.loc[m, "status"] = "approved"
    intake.to_csv(INTAKE, index=False)
    append(FILED, filed)
    b = read(BATCHES)
    m = b["batch"] == batch
    b.loc[m, ["approved_by", "approved_at"]] = [user, now()]
    b.loc[m, "approved"] = str(int(b.loc[m, "approved"].replace("", "0").astype(int).iloc[0]) + len(filed))
    b.to_csv(BATCHES, index=False)
    return len(filed)


def can_undo(batch: str) -> bool:
    """Return whether the batch's approval can still be undone."""
    b = read(BATCHES)
    row = b[b["batch"] == batch]
    if row.empty or not row.iloc[0]["approved_at"]:
        return False
    return datetime.now() - datetime.strptime(row.iloc[0]["approved_at"], "%Y-%m-%d %H:%M") < timedelta(minutes=UNDO_MINUTES)


def undo(batch: str, user: str) -> None:
    """Undo a batch approval: its pages go back to Intake."""
    intake, filed = read(INTAKE), read(FILED)
    back = intake.loc[(intake["batch"] == batch) & (intake["status"] == "approved"), "id"]
    filed[~filed["id"].isin(back)].to_csv(FILED, index=False)
    intake.loc[intake["id"].isin(back), "status"] = "waiting"
    intake.to_csv(INTAKE, index=False)
    b = read(BATCHES)
    m = b["batch"] == batch
    b.loc[m, ["approved_by", "approved_at", "approved"]] = ["", "", "0"]
    b.to_csv(BATCHES, index=False)
    append(REVIEW_LOG, [{"time": now(), "id": f"batch {batch}", "action": "approval undone", "record_type": "",
                         "by": user, "note": f"{len(back)} pages back to Intake"}])


# Review queue
def queue_view(queue: pd.DataFrame) -> pd.DataFrame:
    """Return the pages waiting for a person: scale-run queue plus flagged uploads, minus those filed."""
    intake = read(INTAKE)
    new = intake[intake["status"] == "to review"].copy() if len(intake) else pd.DataFrame()
    if len(new):
        new["from"] = "batch " + new["batch"]
    q = pd.concat([queue.assign(**{"from": "scale run"}), new], ignore_index=True).fillna("")
    log = read(REVIEW_LOG)
    if len(log):
        done = set(log.loc[log["action"] == "filed", "id"])
        q = q[~q["id"].isin(done)]
        second = set(log.loc[log["action"] == "second look requested", "id"])
        q["second_look"] = q["id"].isin(second)
    else:
        q["second_look"] = False
    return q.reset_index(drop=True)


def file_from_review(row: dict, title: str, iso: str, type_id: str, dept: str, user: str, type_info: dict,
                     dept_name: dict) -> None:
    """File one queue page with the reviewer's choices.

    A record type sets the department; with no record type, the chosen department is used.
    """
    row = dict(row)
    if type_id in type_info:
        row, _ = apply_edit({**row, "record_type": ""}, title, iso, type_id, type_info)
    else:
        row.update({"title": title.strip() or row.get("title", ""), "date_iso": clean_date(iso),
                    "department": dept, "department_name": dept_name.get(dept, dept),
                    "record_type": "", "record_type_name": "", "series": "", "series_name": ""})
        row["date_shown"] = date_shown(row["date_iso"])
    thumb = ROOT / row.get("thumb", "")
    if (not row.get("thumb") or not thumb.exists()) and row.get("image") and (ROOT / row["image"]).exists():
        row["thumb"] = rel(make_thumb(ROOT / row["image"], row["id"]))
    rec = to_record(row, f"Filed from the Review queue by {user}", suggested=False)
    append(FILED, [rec])
    append(REVIEW_LOG, [{"time": now(), "id": row["id"], "action": "filed", "record_type": type_id, "by": user,
                         "note": f"suggested {row.get('suggested_record_type', '') or row.get('record_type', '')}"}])
    if row["id"].startswith("b"):
        i = read(INTAKE)
        i.loc[i["id"] == row["id"], "status"] = "filed"
        i.to_csv(INTAKE, index=False)


def review_action(page_id: str, action: str, user: str, note: str = "") -> None:
    """Log a Review queue action."""
    append(REVIEW_LOG, [{"time": now(), "id": page_id, "action": action, "record_type": "", "by": user, "note": note}])
