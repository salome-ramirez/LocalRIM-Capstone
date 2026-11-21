"""Download the sampled documents again in full, by ID.

The first download kept only a 150-dpi image of each first page and some of the
archive's fields. This fetches the same documents (by ID from the manifest, no new draw)
and keeps the PDF, every metadata field and the first page at the scan's own resolution.
The three documents that never downloaded are not retried.

Writes:
    data/pm_sample/pdf/<id>.pdf              the document
    data/pm_sample/images_native/<id>.png    first page at the scan's resolution
    data/pm_sample/metadata_full.jsonl       every archive field, one JSON line per document
    data/pm_sample/metadata_full.csv         the same, one column per field
    data/pm_sample/pdf_info.csv              resolution, size in pixels, pages, text layer
    results/repull_log.txt                   progress and failures

Order: the OCR test pages, any IDs given with --ids-file, the labeled documents, then the
rest. Documents already in pdf_info.csv are skipped. --metadata-only re-fetches only the
fields, into metadata_full_fixed.csv.

Usage:
    python pipeline/01_sample/repull_pm_sample.py --limit 3
    python pipeline/01_sample/repull_pm_sample.py
    python pipeline/01_sample/repull_pm_sample.py --metadata-only
"""

import argparse
import json
import time
from pathlib import Path

import pandas as pd
import pymupdf as fitz
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

ROOT = Path(__file__).resolve().parents[2]
PM_DIR = ROOT / "data" / "pm_sample"
MANIFEST = PM_DIR / "manifest.csv"
PDF_DIR = PM_DIR / "pdf"
IMG_DIR = PM_DIR / "images_native"
META_OUT = PM_DIR / "metadata_full.csv"
META_LINES = PM_DIR / "metadata_full.jsonl"
FIXED_OUT = PM_DIR / "metadata_full_fixed.csv"
FIXED_LINES = PM_DIR / "metadata_full_fixed.jsonl"
INFO_OUT = PM_DIR / "pdf_info.csv"
SPOTCHECK = ROOT / "results" / "ocr_spotcheck_pm_pages.csv"
LABELS = ROOT / "labels" / "label_pm_labels_salo.csv"
LOG = ROOT / "results" / "repull_log.txt"

API = "https://metadata.idl.ucsf.edu/solr/ltdl3/query"
PDF_URL = "https://www.industrydocuments.ucsf.edu/api/v1/files/{id}/{id}.pdf"
NOT_RETRIEVED = {"qnkl0085", "hzgp0030", "xpxk0085"}   # never downloaded; not retried
MAX_DPI = 600          # safety cap if a PDF reports an odd resolution
FALLBACK_DPI = 300     # used when the original resolution can't be read

# same retry settings as get_pm_sample.py
SESSION = requests.Session()
SESSION.mount("https://", HTTPAdapter(max_retries=Retry(
    total=5, backoff_factor=2, status_forcelist=[429, 500, 502, 503, 504],
    allowed_methods=["GET"])))
SEARCH_TIMEOUT = (15, 60)
DOWNLOAD_TIMEOUT = (15, 180)
API_HEADERS = {"User-Agent": "LocalRIM-research-sample/1.0"}


def log(msg):
    """Append a timestamped line to the run log."""
    LOG.parent.mkdir(exist_ok=True)
    with LOG.open("a", encoding="utf-8") as f:
        f.write(f"{time.strftime('%H:%M:%S')} {msg}\n")


def as_text(v):
    """Join a list field into one string."""
    return "; ".join(str(x) for x in v) if isinstance(v, list) else ("" if v is None else str(v))


def fetch_metadata(doc_id):
    """Return every field the archive holds for one document."""
    r = SESSION.get(API, params={"q": f"id:{doc_id}", "wt": "json", "fl": "*", "rows": 1},
                    headers=API_HEADERS, timeout=SEARCH_TIMEOUT)
    r.raise_for_status()
    time.sleep(0.3)   # be polite to the archive's server
    docs = r.json()["response"]["docs"]
    if not docs:
        raise ValueError("no metadata found")
    return {k: as_text(v) for k, v in docs[0].items()}


def fetch_pdf(doc_id):
    """Download a document's PDF (same headers and PDF check as get_pm_sample.py)."""
    headers = {
        "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                       "(KHTML, like Gecko) Chrome/130.0 Safari/537.36"),
        "Referer": f"https://www.industrydocuments.ucsf.edu/docs/{doc_id}",
        "Accept": "application/pdf,*/*",
    }
    r = SESSION.get(PDF_URL.format(id=doc_id), headers=headers, timeout=DOWNLOAD_TIMEOUT)
    r.raise_for_status()
    if not r.content.startswith(b"%PDF-"):
        raise ValueError(f"not a PDF (got {r.headers.get('Content-Type', 'unknown type')})")
    return r.content


def native_dpi(page):
    """Resolution of the scan stored on this page: image pixels / page width in inches."""
    best = None
    for img in page.get_images(full=True):
        width_px = img[2]
        if width_px and page.rect.width:
            dpi = width_px / (page.rect.width / 72)
            best = max(best or 0, dpi)
    return round(best) if best else None


def process(doc_id):
    """Fetch one document's fields and PDF, render its first page and return its facts."""
    meta = fetch_metadata(doc_id)

    pdf_path = PDF_DIR / f"{doc_id}.pdf"
    if not pdf_path.exists():
        pdf_path.write_bytes(fetch_pdf(doc_id))

    pdf = fitz.open(pdf_path)
    page = pdf[0]
    dpi = native_dpi(page)
    render_dpi = min(dpi or FALLBACK_DPI, MAX_DPI)
    pix = page.get_pixmap(dpi=render_dpi)
    pix.save(IMG_DIR / f"{doc_id}.png")
    text_layer = page.get_text().strip()

    info = {
        "id": doc_id,
        "pdf_pages": pdf.page_count,
        "native_dpi": dpi or "",
        "render_dpi": render_dpi,
        "width_px": pix.width,
        "height_px": pix.height,
        "has_text_layer": bool(text_layer),
        "text_layer_chars": len(text_layer),
        "pdf_bytes": pdf_path.stat().st_size,
    }
    pdf.close()
    return meta, info


def append(path, row):
    """Append one row to a CSV, writing the header the first time."""
    pd.DataFrame([row]).to_csv(path, mode="a", header=not path.exists(), index=False)


def save_meta_line(path, meta):
    """Save one document's fields by name (safe when documents have different fields)."""
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(meta, ensure_ascii=False) + "\n")


def read_meta_lines(path):
    """Read the saved metadata lines."""
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_meta_csv(lines_path, csv_path):
    """One row per document (newest save wins), one column for every field seen."""
    rows = read_meta_lines(lines_path)
    if not rows:
        return 0
    df = pd.DataFrame(rows).drop_duplicates("id", keep="last")
    df = df[["id"] + [c for c in df.columns if c != "id"]]
    df.to_csv(csv_path, index=False)
    return len(df)


def metadata_only(manifest, limit):
    """Re-fetch only the archive fields for every document, into metadata_full_fixed.csv."""
    ids = order(manifest, None)
    done = {r["id"] for r in read_meta_lines(FIXED_LINES)}
    todo = [i for i in ids if i not in done]
    if limit:
        todo = todo[:limit]
    print(f"Metadata only. Already done: {len(done)}. To do now: {len(todo)}.")
    log(f"--- metadata-only run {time.ctime()} | done {len(done)} | to do {len(todo)}")

    failures_in_a_row = 0
    for n, doc_id in enumerate(todo, 1):
        try:
            save_meta_line(FIXED_LINES, fetch_metadata(doc_id))
            failures_in_a_row = 0
        except KeyboardInterrupt:
            print("Stopped. Finished documents are saved; run again to continue.")
            break
        except Exception as e:
            log(f"metadata failed {doc_id}: {str(e)[:200]}")
            failures_in_a_row += 1
            if failures_in_a_row >= 10:
                print("Stopped: 10 failures in a row (connection problem?). See outputs\\repull_log.txt")
                break
            continue
        if n % 50 == 0 or n == len(todo):
            print(f"{n}/{len(todo)}", flush=True)

    written = write_meta_csv(FIXED_LINES, FIXED_OUT)
    print(f"Wrote {written} documents to {FIXED_OUT}")


def order(manifest, ids_file):
    """Return the document IDs in download order."""
    ids = manifest["id"].tolist()
    first = []
    if SPOTCHECK.exists():
        first += pd.read_csv(SPOTCHECK, dtype=str)["id"].tolist()
    if ids_file:
        first += pd.read_csv(ids_file, dtype=str)["id"].tolist()
    if LABELS.exists():
        labeled = set(pd.read_csv(LABELS, dtype=str)["doc_id"])
        first += [i for i in ids if i in labeled]
    seen, out = set(), []
    for i in first + ids:
        if i in ids and i not in seen and i not in NOT_RETRIEVED:
            seen.add(i)
            out.append(i)
    return out


def main():
    """Download every document not yet done."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="only do this many documents")
    parser.add_argument("--ids-file", default=None, help="CSV with an 'id' column to pull first")
    parser.add_argument("--metadata-only", action="store_true",
                        help="re-fetch archive fields only, into metadata_full_fixed.csv")
    args = parser.parse_args()

    if args.metadata_only:
        metadata_only(pd.read_csv(MANIFEST, dtype=str), args.limit)
        return

    PDF_DIR.mkdir(parents=True, exist_ok=True)
    IMG_DIR.mkdir(parents=True, exist_ok=True)
    manifest = pd.read_csv(MANIFEST, dtype=str)

    done = set()
    if INFO_OUT.exists():
        done = set(pd.read_csv(INFO_OUT, dtype=str)["id"])
    todo = [i for i in order(manifest, args.ids_file) if i not in done]
    if args.limit:
        todo = todo[: args.limit]
    print(f"Already done: {len(done)}. To do now: {len(todo)}.")
    log(f"--- run {time.ctime()} | done {len(done)} | to do {len(todo)}")

    start, failures_in_a_row = time.time(), 0
    for n, doc_id in enumerate(todo, 1):
        try:
            meta, info = process(doc_id)
            save_meta_line(META_LINES, meta)
            append(INFO_OUT, info)
            failures_in_a_row = 0
        except KeyboardInterrupt:
            print("Stopped. Finished documents are saved; run again to continue.")
            return
        except Exception as e:   # one bad document should not stop the run
            log(f"failed {doc_id}: {str(e)[:200]}")
            failures_in_a_row += 1
            if failures_in_a_row >= 10:
                print("Stopped: 10 failures in a row (connection problem?). See outputs\\repull_log.txt")
                return
            continue

        if n % 10 == 0 or n == len(todo):
            per_doc = (time.time() - start) / n
            left = per_doc * (len(todo) - n) / 60
            print(f"{n}/{len(todo)} | {per_doc:.1f} s per document | about {left:.0f} min left "
                  f"| last: {doc_id} at {info['native_dpi'] or '?'} dpi", flush=True)

    write_meta_csv(META_LINES, META_OUT)
    log(f"done: {len(todo)} processed")
    print("Finished. See data\\pm_sample\\pdf_info.csv and outputs\\repull_log.txt")


if __name__ == "__main__":
    main()
