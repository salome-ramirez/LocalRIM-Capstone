"""OCR engine A: Tesseract on the 150-dpi first pages.

Reads every first page in data/pm_sample/images in manifest order, reads it with Tesseract
(English and German), removes the archive's source footer and saves the text. Rotation:
each page is read as scanned; if Tesseract's orientation check says to turn it, the turned
page is read too and the more confident reading is kept. Pages that never downloaded are
skipped.

Writes:
    data/pm_sample/ocr/<id>.txt          text
    data/pm_sample/ocr/ocr_results.csv   path, rotated, suggested, words, mean_conf, chars, seconds, error

Resumable: pages in ocr_results.csv are skipped and failed pages retried.
--redo-rotated re-reads pages an earlier run turned on the orientation check alone.

Usage:
    python -m pipeline.02_ocr.run_ocr --limit 5
    python -m pipeline.02_ocr.run_ocr
"""

import argparse
import time
from pathlib import Path

import pandas as pd
import pytesseract
from PIL import Image

from pipeline.common import Progress, ROOT as REPO

# Settings
PM_DIR = REPO / "data" / "pm_sample"
IMAGE_DIR = PM_DIR / "images"
MANIFEST = PM_DIR / "manifest.csv"
OCR_DIR = PM_DIR / "ocr"
RESULTS = OCR_DIR / "ocr_results.csv"

# Where the Windows installer puts Tesseract; used so PATH doesn't matter
WINDOWS_TESSERACT = Path(r"C:\Program Files\Tesseract-OCR\tesseract.exe")
MAX_ERRORS_IN_A_ROW = 5   # stop the run if pages keep failing (something is wrong, not the page)
UPSCALE = 1           # PM pages are rendered at 150 dpi (about 1280 x 1650)
LANG = "eng+deu"      # English + German; needs the deu language pack
CONFIG = "--psm 3"    # automatic page layout
# downloads that returned the archive's web page instead of the scan
NOT_RETRIEVED = {"qnkl0085", "hzgp0030", "xpxk0085"}
# the archive's source line on every page is not part of the record
FOOTER_MARKERS = ("industrydocuments", "Source: https")


def page_order() -> list[str]:
    """Return all sample pages in manifest order, minus pages not retrieved."""
    manifest = pd.read_csv(MANIFEST, dtype=str)
    pages = []
    for doc_id, image in zip(manifest["id"], manifest["image"]):
        if doc_id in NOT_RETRIEVED:
            continue
        rel = Path(image).name            # manifest stores "images/<id>.png"
        if (IMAGE_DIR / rel).exists():
            pages.append(rel)
    return pages


def strip_footer(text: str) -> str:
    """Drop the archive's source line."""
    return "\n".join(line for line in text.splitlines()
                     if not any(m in line for m in FOOTER_MARKERS))


def suggested_rotation(image: Image.Image) -> int:
    """Ask Tesseract's orientation check which way the page should turn (0 if it can't tell)."""
    try:
        osd = pytesseract.image_to_osd(image, output_type=pytesseract.Output.DICT)
        return int(osd.get("rotate", 0))
    except KeyboardInterrupt:
        raise      # Ctrl+C still stops the run
    except Exception:
        return 0   # too little text to tell, or an odd reply


def read_words(image: Image.Image) -> tuple[str, int, float]:
    """Read one image; return the text, word count and mean confidence."""
    # Plain dictionary output: every word stays text (a table would turn
    # number-only words like "1994" into numbers and break the join)
    d = pytesseract.image_to_data(image, lang=LANG, config=CONFIG,
                                  output_type=pytesseract.Output.DICT)
    confs, lines = [], {}
    for i, word in enumerate(d["text"]):
        word = str(word).strip()
        conf = float(d["conf"][i])
        if conf < 0 or not word:      # keep real words only
            continue
        confs.append(conf)
        # Rebuild the text line by line from the words Tesseract found
        key = (d["block_num"][i], d["par_num"][i], d["line_num"][i])
        lines.setdefault(key, []).append(word)
    text = "\n".join(" ".join(lines[k]) for k in sorted(lines))
    mean_conf = round(sum(confs) / len(confs), 1) if confs else 0.0
    return text, len(confs), mean_conf


def ocr_page(rel_path: str) -> dict:
    """Read one page, also turned when the orientation check suggests it; keep the more confident reading."""
    start = time.time()
    image = Image.open(IMAGE_DIR / rel_path).convert("L")
    if UPSCALE != 1:
        image = image.resize((image.width * UPSCALE, image.height * UPSCALE), Image.LANCZOS)

    # Read the page as it is. If the orientation check says to turn it, read the
    # turned version too and keep whichever Tesseract is more confident about.
    # (Trusting the check alone turned many upright pages upside down.)
    text, words, conf = read_words(image)
    suggested = suggested_rotation(image)
    rotated = 0
    if suggested:
        t2, w2, c2 = read_words(image.rotate(-suggested, expand=True))
        if c2 > conf:
            text, words, conf, rotated = t2, w2, c2, suggested

    text = strip_footer(text)
    out = OCR_DIR / Path(rel_path).with_suffix(".txt")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")

    return {"path": rel_path, "rotated": rotated, "suggested": suggested, "words": words,
            "mean_conf": conf, "chars": len(text),
            "seconds": round(time.time() - start, 2), "error": ""}


def main():
    """Read every page not yet in the results."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="only do this many new pages")
    parser.add_argument("--redo-rotated", action="store_true",
                        help="redo pages that were turned (kept from the Nov 6 pilot fix)")
    args = parser.parse_args()

    if WINDOWS_TESSERACT.exists():
        pytesseract.pytesseract.tesseract_cmd = str(WINDOWS_TESSERACT)
    try:
        print(f"Using Tesseract {pytesseract.get_tesseract_version()}")
    except Exception:
        print("Can't find Tesseract. Check it is installed (tesseract --version).")
        return

    OCR_DIR.mkdir(parents=True, exist_ok=True)
    done = set()
    if RESULTS.exists():
        # Pages that failed last time are removed and tried again
        results = pd.read_csv(RESULTS, dtype=str, keep_default_na=False)
        results = results[results["error"] == ""]
        if args.redo_rotated:
            # The first run trusted the orientation check alone; redo those pages
            results = results[results["rotated"] == "0"]
        if "suggested" not in results.columns:   # files from the first run lack this column
            results.insert(2, "suggested", results["rotated"])
        results.to_csv(RESULTS, index=False)
        done = set(results["path"])
    todo = [p for p in page_order() if p not in done]
    if args.limit:
        todo = todo[: args.limit]
    print(f"Already done: {len(done)}. To do now: {len(todo)}.")

    progress = Progress(len(todo))
    errors_in_a_row = 0
    for i, rel_path in enumerate(todo, 1):
        try:
            row = ocr_page(rel_path)
            errors_in_a_row = 0
        except KeyboardInterrupt:
            print("Stopped. Finished pages are saved; run again to continue.")
            return
        except Exception as e:  # one bad page should not stop the run
            row = {"path": rel_path, "rotated": 0, "suggested": 0, "words": 0, "mean_conf": 0.0,
                   "chars": 0, "seconds": 0.0, "error": str(e)[:200]}
            errors_in_a_row += 1
        # Save after every page, so stopping never loses work
        pd.DataFrame([row]).to_csv(RESULTS, mode="a", header=not RESULTS.exists(), index=False)
        if errors_in_a_row >= MAX_ERRORS_IN_A_ROW:
            print(f"Stopped: {MAX_ERRORS_IN_A_ROW} pages failed in a row. Last error: {row['error']}")
            return
        progress(i)

    print(f"Finished. Results: {RESULTS}")


if __name__ == "__main__":
    main()
