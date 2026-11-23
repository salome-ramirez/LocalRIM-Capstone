"""Engine E: copy the archive's own text layer out of each PDF.

The library embedded its own OCR text in each PDF. This copies the first page's text and
word positions; no OCR is run. Used as a reference, not as a candidate engine.

Writes:
    data/pm_sample/ocr_ucsf/<id>.txt          text, footer removed
    data/pm_sample/ocr_ucsf/boxes/<id>.tsv    one row per word with its box (PDF points)
    data/pm_sample/ocr_ucsf/text_results.csv  id, words, chars, error

Usage:
    python -m pipeline.02_ocr.extract_text_layer
"""


import pandas as pd
import pymupdf as fitz

from pipeline.common import ROOT as REPO

PM_DIR = REPO / "data" / "pm_sample"
PDF_DIR = PM_DIR / "pdf"
OUT_DIR = PM_DIR / "ocr_ucsf"
BOX_DIR = OUT_DIR / "boxes"
RESULTS = OUT_DIR / "text_results.csv"
FOOTER_MARKERS = ("industrydocuments", "Source: https")


def main():
    """Copy the text layer of every first page."""
    BOX_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    pdfs = sorted(PDF_DIR.glob("*.pdf"))
    for n, pdf_path in enumerate(pdfs, 1):
        doc_id = pdf_path.stem
        try:
            with fitz.open(pdf_path) as pdf:
                page = pdf[0]
                text = page.get_text("text")
                words = page.get_text("words")   # x0, y0, x1, y1, word, block, line, word_no
            text = "\n".join(l for l in text.splitlines()
                             if l.strip() and not any(m in l for m in FOOTER_MARKERS))
            (OUT_DIR / f"{doc_id}.txt").write_text(text, encoding="utf-8")
            boxes = pd.DataFrame(words, columns=["x0", "y0", "x1", "y1", "text", "block", "line", "word_no"])
            boxes.to_csv(BOX_DIR / f"{doc_id}.tsv", sep="\t", index=False)
            rows.append({"id": doc_id, "words": len(text.split()), "chars": len(text), "error": ""})
        except Exception as e:
            rows.append({"id": doc_id, "words": 0, "chars": 0, "error": str(e)[:200]})
        if n % 200 == 0:
            print(f"  {n}/{len(pdfs)}", flush=True)
    pd.DataFrame(rows).to_csv(RESULTS, index=False)
    res = pd.DataFrame(rows)
    print(f"Done: {len(res)} PDFs, {int((res['words'] > 0).sum())} with text, "
          f"{int((res['error'] != '').sum())} errors. Results: {RESULTS}")


if __name__ == "__main__":
    main()
