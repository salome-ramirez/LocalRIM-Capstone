"""Download the sample's PDFs from the UCSF Industry Documents Library and render their first pages.

Reads the document IDs from data/pm_sample/pdf_info.csv and saves, for each one:

    data/pm_sample/pdf/<id>.pdf             the document, as published by the library
    data/pm_sample/images_native/<id>.png   its first page at the scan's own resolution

The page is rendered at the resolution recorded in data/pm_sample/pdf_info.csv, so the
images match the ones the classifier and the archive were built from. Files that already
exist are skipped, so the script can be stopped and run again.

Usage:
    python -m pipeline.01_sample.download_pdfs
    python -m pipeline.01_sample.download_pdfs --limit 5     # try a few documents first
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd
import pymupdf as fitz
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from pipeline.common import ROOT

DATA = ROOT / "data" / "pm_sample"
PDF_DIR = DATA / "pdf"
IMG_DIR = DATA / "images_native"
PDF_URL = "https://www.industrydocuments.ucsf.edu/api/v1/files/{id}/{id}.pdf"
PAUSE = 0.3                      # seconds between downloads
DOWNLOAD_TIMEOUT = (15, 180)     # connect, read

SESSION = requests.Session()
SESSION.mount("https://", HTTPAdapter(max_retries=Retry(
    total=5, backoff_factor=2, status_forcelist=[429, 500, 502, 503, 504], allowed_methods=["GET"])))


def fetch_pdf(doc_id: str) -> bytes:
    """Return the PDF for one document ID; raise if the server sends anything else."""
    # The file server rejects requests without browser-like headers (403).
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


def render_first_page(pdf_path: Path, png_path: Path, dpi: int) -> None:
    """Save the first page of a PDF as a PNG at the given resolution."""
    with fitz.open(pdf_path) as pdf:
        pdf[0].get_pixmap(dpi=dpi).save(png_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--limit", type=int, help="download at most this many documents")
    args = parser.parse_args()

    info = pd.read_csv(DATA / "pdf_info.csv", dtype={"id": str})
    ids, render_dpi = info["id"].tolist(), info.set_index("id")["render_dpi"]
    if args.limit:
        ids = ids[: args.limit]
    PDF_DIR.mkdir(parents=True, exist_ok=True)
    IMG_DIR.mkdir(parents=True, exist_ok=True)

    failed = []
    for n, doc_id in enumerate(ids, 1):
        pdf_path, png_path = PDF_DIR / f"{doc_id}.pdf", IMG_DIR / f"{doc_id}.png"
        try:
            if not pdf_path.exists():
                pdf_path.write_bytes(fetch_pdf(doc_id))
                time.sleep(PAUSE)
            if not png_path.exists():
                render_first_page(pdf_path, png_path, int(render_dpi[doc_id]))
        except Exception as e:
            failed.append(doc_id)
            print(f"{doc_id}: failed ({str(e)[:120]})", flush=True)
        if n % 50 == 0 or n == len(ids):
            print(f"{n}/{len(ids)} documents", flush=True)

    print(f"Done: {len(ids) - len(failed)} of {len(ids)} documents in {PDF_DIR.relative_to(ROOT)}")
    if failed:
        print("Failed (run the script again to retry): " + ", ".join(failed))


if __name__ == "__main__":
    main()
