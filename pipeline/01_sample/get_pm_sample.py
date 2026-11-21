"""Draw the random sample of Philip Morris documents and save each first page.

Searches the UCSF Industry Documents Library for public, scanned documents in the
Philip Morris Records collection that are not also filed under another company's
collection, and picks documents at random (fixed seed): a random three-letter ID slice,
then a random document in it, with slices kept in proportion to their size. Duplicates
(same title, date and page count) are dropped. For each document it saves the first page
as a 150-dpi PNG and a manifest row with the archive's metadata.

Writes:
    data/pm_sample/images/<id>.png    first page, 150 dpi
    data/pm_sample/manifest.csv       id, title, documentdate, type, pages, area, bates, collection, image
    results/pm_sample_log.txt         progress and failures

Usage:
    python pipeline/01_sample/get_pm_sample.py --limit 5    # try a few documents
    python pipeline/01_sample/get_pm_sample.py              # the full draw (resumable)

API: https://www.industrydocuments.ucsf.edu/research-tools/api/
"""

import argparse
import random
import time
from pathlib import Path

import pymupdf as fitz  # PyMuPDF
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "data" / "pm_sample"
IMG_DIR = OUT_DIR / "images"
MANIFEST = OUT_DIR / "manifest.csv"
LOG = ROOT / "results" / "pm_sample_log.txt"

API = "https://metadata.idl.ucsf.edu/solr/ltdl3/query"
QUERY = ('collection:"Philip Morris Records" AND availability:public '
         'AND format:"scanned hardcopy"')
FIELDS = "id,title,documentdate,type,pages,area,bates,collection,format"
# Download address used by the archive's own viewer; it redirects (302) to the file.
PDF_URL = "https://www.industrydocuments.ucsf.edu/api/v1/files/{id}/{id}.pdf"

# a document also filed under one of these collections is dropped
OTHER_COMPANIES = ["Reynolds", "Lorillard", "Brown & Williamson", "Tobacco Institute",
                   "Council for Tobacco", "American Tobacco", "Liggett"]

TARGET = 1400   # 400 for labeling + 1,000 for the scale run
SEED = 44
DPI = 150


def log(msg):
    """Append a line to the run log."""
    LOG.parent.mkdir(exist_ok=True)
    with LOG.open("a", encoding="utf-8") as f:
        f.write(msg + "\n")


HEADERS = {"User-Agent": "LocalRIM-research-sample/1.0"}

# One shared session with retries and separate connect/read timeouts, so a stalled
# connection is dropped and retried instead of hanging the run.
# requests sessions: https://requests.readthedocs.io/en/latest/user/advanced/#session-objects
# urllib3 Retry:     https://urllib3.readthedocs.io/en/stable/reference/urllib3.util.html
SESSION = requests.Session()
SESSION.mount("https://", HTTPAdapter(max_retries=Retry(
    total=5, backoff_factor=2, status_forcelist=[429, 500, 502, 503, 504],
    allowed_methods=["GET"])))
SEARCH_TIMEOUT = (15, 60)     # seconds to connect, seconds to wait for an answer
DOWNLOAD_TIMEOUT = (15, 120)
LETTERS = "abcdefghijklmnopqrstuvwxyz"
SLICE_CAP = 3000  # for fair weighting: a slice with n docs is kept with chance n / SLICE_CAP
                  # (above the largest slice seen)


def search(query, start=0, rows=1):
    """Run one archive search and return its response."""
    r = SESSION.get(API, params={"q": query, "wt": "json", "fl": FIELDS,
                                 "start": start, "rows": rows},
                    headers=HEADERS, timeout=SEARCH_TIMEOUT)
    r.raise_for_status()
    time.sleep(0.3)  # be polite to the archive's server
    return r.json()["response"]


def random_doc(rng):
    """Pick a random three-letter ID slice, then a random document in it.

    The server refuses deep jumps into the full list (403), but slices are short. Slices
    are kept in proportion to their size, so every document has the same chance.
    """
    prefix = "".join(rng.choice(LETTERS) for _ in range(3))
    query = f"{QUERY} AND id:{prefix}*"
    n = search(query, rows=0)["numFound"]
    if n > SLICE_CAP:
        log(f"WARNING: slice {prefix} has {n} docs (over {SLICE_CAP}); weighting not exact for it")
    if n == 0 or rng.random() > n / SLICE_CAP:
        return None
    docs = search(query, start=rng.randrange(min(n, SLICE_CAP)))["docs"]
    return docs[0] if docs else None


def is_pm_only(doc):
    """Return whether a document is filed only under Philip Morris Records."""
    cols = doc.get("collection", [])
    return "Philip Morris Records" in cols and not any(
        other in c for c in cols for other in OTHER_COMPANIES)


def first_page_png(doc_id):
    """Download a document's PDF and save its first page as a 150-dpi PNG."""
    url = PDF_URL.format(id=doc_id)
    # The download server refuses plain scripts (403); send the headers a browser sends.
    browser_headers = {
        "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                       "(KHTML, like Gecko) Chrome/130.0 Safari/537.36"),
        "Referer": f"https://www.industrydocuments.ucsf.edu/docs/{doc_id}",
        "Accept": "application/pdf,*/*",
    }
    r = SESSION.get(url, headers=browser_headers, timeout=DOWNLOAD_TIMEOUT)
    r.raise_for_status()
    # Only a real PDF counts: some documents answer with the archive's "enable JavaScript"
    # page instead. A PDF starts with "%PDF-"; anything else is refused and logged.
    if not r.content.startswith(b"%PDF-"):
        raise ValueError(f"not a PDF (got {r.headers.get('Content-Type', 'unknown type')})")
    pdf = fitz.open(stream=r.content, filetype="pdf")
    pix = pdf[0].get_pixmap(dpi=DPI)
    pix.save(IMG_DIR / f"{doc_id}.png")


def as_text(v):
    """Join a list field into one string."""
    return "; ".join(v) if isinstance(v, list) else (v or "")


def main():
    """Draw documents until the target is reached, saving the manifest as it goes."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=TARGET)
    args = parser.parse_args()

    IMG_DIR.mkdir(parents=True, exist_ok=True)
    rows = pd.read_csv(MANIFEST, dtype=str).to_dict("records") if MANIFEST.exists() else []
    seen_ids = {r["id"] for r in rows}
    seen_keys = {(r["title"], r["documentdate"], r["pages"]) for r in rows}

    total = search(QUERY, rows=0)["numFound"]
    log(f"--- run {time.ctime()} | matches in archive: {total} | target {args.limit}")
    rng = random.Random(SEED)
    tries = 0
    failures = 0

    while len(rows) < args.limit and tries < args.limit * 200:
        tries += 1
        if tries % 200 == 0:   # progress output while earlier picks are replayed
            print(f"  ...searching: {tries} tries, {len(rows)} documents so far", flush=True)
        try:
            doc = random_doc(rng)
        except Exception as e:
            failures += 1
            log(f"search failed: {e}")
            if failures >= 10:
                log("stopped: 10 search failures")
                break
            continue
        if doc is None:
            continue
        doc_id = doc["id"]
        key = (as_text(doc.get("title")), as_text(doc.get("documentdate")), str(doc.get("pages", "")))
        if doc_id in seen_ids or key in seen_keys or not is_pm_only(doc):
            continue
        try:
            first_page_png(doc_id)
        except Exception as e:
            log(f"download failed {doc_id}: {e}")
            continue
        seen_ids.add(doc_id)
        seen_keys.add(key)
        rows.append({
            "id": doc_id,
            "title": key[0],
            "documentdate": key[1],
            "pages": key[2],
            "type": as_text(doc.get("type")),
            "area": as_text(doc.get("area")),
            "bates": as_text(doc.get("bates")),
            "collection": as_text(doc.get("collection")),
            "image": f"images/{doc_id}.png",
        })
        pd.DataFrame(rows).to_csv(MANIFEST, index=False)  # saved every time: resumable
        print(f"  {len(rows)} / {args.limit}  {doc_id}")

    log(f"done: {len(rows)} documents saved after {tries} tries")
    print(f"Saved {len(rows)} documents to data\\pm_sample\\  (details: outputs\\pm_sample_log.txt)")


if __name__ == "__main__":
    main()
