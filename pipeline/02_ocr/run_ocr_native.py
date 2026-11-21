"""OCR engine B: Tesseract on the first pages at the scan's own resolution.

Same settings as engine A (run_ocr.py), so A and B differ only in image resolution.
Also saves every word's position and confidence.

Writes:
    data/pm_sample/ocr_tess_native/<id>.txt          text, footer removed
    data/pm_sample/ocr_tess_native/boxes/<id>.tsv    one row per word with its box
    data/pm_sample/ocr_tess_native/ocr_results.csv   id, rotated, words, mean_conf, chars, seconds, error

Resumable: pages in ocr_results.csv are skipped.

Usage:
    python pipeline/02_ocr/run_ocr_native.py --limit 5
    python pipeline/02_ocr/run_ocr_native.py
"""

import argparse
import time
from pathlib import Path

import pandas as pd
import pytesseract
from PIL import Image

REPO = Path(__file__).resolve().parents[2]
PM_DIR = REPO / "data" / "pm_sample"
IMAGE_DIR = PM_DIR / "images_native"
MANIFEST = PM_DIR / "manifest.csv"
OUT_DIR = PM_DIR / "ocr_tess_native"
BOX_DIR = OUT_DIR / "boxes"
RESULTS = OUT_DIR / "ocr_results.csv"

WINDOWS_TESSERACT = Path(r"C:\Program Files\Tesseract-OCR\tesseract.exe")
LANG = "eng+deu"
CONFIG = "--psm 3"
FOOTER_MARKERS = ("industrydocuments", "Source: https")
NOT_RETRIEVED = {"qnkl0085", "hzgp0030", "xpxk0085"}   # never downloaded
MAX_ERRORS_IN_A_ROW = 5


def read_words(image):
    """Read one image; return the text, word count, mean confidence and the word table."""
    d = pytesseract.image_to_data(image, lang=LANG, config=CONFIG,
                                  output_type=pytesseract.Output.DICT)
    rows, lines = [], {}
    for i, word in enumerate(d["text"]):
        word = str(word).strip()
        conf = float(d["conf"][i])
        if conf < 0 or not word:
            continue
        key = (d["block_num"][i], d["par_num"][i], d["line_num"][i])
        lines.setdefault(key, []).append(word)
        rows.append({"block": d["block_num"][i], "par": d["par_num"][i], "line": d["line_num"][i],
                     "left": d["left"][i], "top": d["top"][i], "width": d["width"][i],
                     "height": d["height"][i], "conf": conf, "text": word})
    text = "\n".join(" ".join(lines[k]) for k in sorted(lines))
    mean_conf = round(sum(r["conf"] for r in rows) / len(rows), 1) if rows else 0.0
    return text, len(rows), mean_conf, rows


def suggested_rotation(image):
    """Ask Tesseract's orientation check which way the page should turn (0 if it can't tell)."""
    try:
        return int(pytesseract.image_to_osd(image, output_type=pytesseract.Output.DICT).get("rotate", 0))
    except KeyboardInterrupt:
        raise
    except Exception:
        return 0


def ocr_page(doc_id):
    """Read one page, keeping the more confident orientation, and save its text and boxes."""
    start = time.time()
    image = Image.open(IMAGE_DIR / f"{doc_id}.png").convert("L")
    text, words, conf, rows = read_words(image)
    suggested = suggested_rotation(image)
    rotated = 0
    if suggested:
        t2, w2, c2, r2 = read_words(image.rotate(-suggested, expand=True))
        if c2 > conf:
            text, words, conf, rows, rotated = t2, w2, c2, r2, suggested

    text = "\n".join(l for l in text.splitlines() if not any(m in l for m in FOOTER_MARKERS))
    (OUT_DIR / f"{doc_id}.txt").write_text(text, encoding="utf-8")
    # Boxes are in the coordinates of the image that was read (turned by "rotated" degrees)
    boxes = pd.DataFrame(rows)
    if len(boxes):
        boxes = boxes[~boxes["text"].str.contains("|".join(FOOTER_MARKERS), regex=True)]
    boxes.to_csv(BOX_DIR / f"{doc_id}.tsv", sep="\t", index=False)

    return {"id": doc_id, "rotated": rotated, "suggested": suggested, "words": words,
            "mean_conf": conf, "chars": len(text), "width_px": image.width,
            "height_px": image.height, "seconds": round(time.time() - start, 2), "error": ""}


def main():
    """Read every page not yet in the results."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    if WINDOWS_TESSERACT.exists():
        pytesseract.pytesseract.tesseract_cmd = str(WINDOWS_TESSERACT)
    print(f"Using Tesseract {pytesseract.get_tesseract_version()}")

    BOX_DIR.mkdir(parents=True, exist_ok=True)
    done = set()
    if RESULTS.exists():
        res = pd.read_csv(RESULTS, dtype=str, keep_default_na=False)
        res = res[res["error"] == ""]
        res.to_csv(RESULTS, index=False)
        done = set(res["id"])

    ids = pd.read_csv(MANIFEST, dtype=str)["id"]
    todo = [i for i in ids if i not in NOT_RETRIEVED and i not in done
            and (IMAGE_DIR / f"{i}.png").exists()]
    if args.limit:
        todo = todo[: args.limit]
    print(f"Already done: {len(done)}. To do now: {len(todo)}.")

    start, errors = time.time(), 0
    for n, doc_id in enumerate(todo, 1):
        try:
            row = ocr_page(doc_id)
            errors = 0
        except KeyboardInterrupt:
            print("Stopped. Finished pages are saved; run again to continue.")
            return
        except Exception as e:
            row = {"id": doc_id, "rotated": 0, "suggested": 0, "words": 0, "mean_conf": 0.0,
                   "chars": 0, "width_px": 0, "height_px": 0, "seconds": 0.0, "error": str(e)[:200]}
            errors += 1
        pd.DataFrame([row]).to_csv(RESULTS, mode="a", header=not RESULTS.exists(), index=False)
        if errors >= MAX_ERRORS_IN_A_ROW:
            print(f"Stopped: {MAX_ERRORS_IN_A_ROW} pages failed in a row. Last error: {row['error']}")
            return
        if n % 10 == 0 or n == len(todo):
            per = (time.time() - start) / n
            print(f"{n}/{len(todo)} pages | {per:.1f} s per page | about {per * (len(todo) - n) / 60:.0f} min left",
                  flush=True)
    print(f"Finished. Results: {RESULTS}")


if __name__ == "__main__":
    main()
