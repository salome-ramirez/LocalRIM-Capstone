"""OCR engine C: PaddleOCR on the first pages at the scan's own resolution.

- models: the installed PaddleOCR's default pipeline (PaddleOCR 3.7: PP-OCRv6_medium_det,
  PP-OCRv6_medium_rec, PP-LCNet_x1_0_doc_ori and PP-LCNet_x1_0_textline_ori)
- page orientation: each page is read as scanned; if the orientation model says to turn
  it, the turned reading is also made and the more confident of the two is kept
- text-line orientation on (reads rotated stamps and folder tabs), unwarping off, GPU
- footer: the archive's source footer is removed, by its exact words anywhere on the page
  and by footer-like text in the bottom FOOTER_BAND, even when misread; box positions
  account for PaddleOCR shrinking pages larger than MAX_SIDE

The app's OCR helper (app/localrim/ocr_worker.py) uses the same reading functions.
Runs in .venv-ocr (Python 3.12).

Writes data/pm_sample/ocr_paddle/<id>.txt, boxes/<id>.tsv, ocr_results.csv and
run_info.json. Order: the OCR test pages first, then the rest; resumable.

Usage (in .venv-ocr):
    python -m pipeline.02_ocr.run_ocr_paddle --limit 5
    python -m pipeline.02_ocr.run_ocr_paddle
    python -m pipeline.02_ocr.run_ocr_paddle --refilter    # re-apply the footer rule to saved boxes
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import pandas as pd
from PIL import Image

from pipeline.common import Progress, ROOT, resume_csv

DATA = ROOT / "data" / "pm_sample"
IMAGE_DIR = DATA / "images_native"
OUT_DIR = DATA / "ocr_paddle"
BOX_DIR = OUT_DIR / "boxes"
RESULTS = OUT_DIR / "ocr_results.csv"
RUN_INFO = OUT_DIR / "run_info.json"
TEST_PAGES = ROOT / "results" / "ocr_spotcheck_pm_pages.csv"

NOT_RETRIEVED = {"qnkl0085", "hzgp0030", "xpxk0085"}   # never downloaded
FOOTER_BAND = 0.05      # bottom 5% of the page
FOOTER_WORDS = re.compile(r"sou|urce|http|ucsf|docs|indus|library|tobacco", re.IGNORECASE)
FOOTER_MARKERS = ("industrydocuments", "Source: http")   # removed anywhere, as for the other engines
MAX_SIDE = 4000         # PaddleOCR shrinks larger pages to this size; boxes are in the shrunk page
MAX_ERRORS_IN_A_ROW = 5

COMMON = {"use_doc_unwarping": False, "use_textline_orientation": True, "device": "gpu"}
AUTO = {**COMMON, "use_doc_orientation_classify": True}     # turns the page if the model says so
PLAIN = {**COMMON, "use_doc_orientation_classify": False}   # reads the page as scanned


def as_dict(res) -> dict:
    """PaddleOCR results behave like dicts; fall back to the JSON form if not."""
    try:
        d = dict(res)
    except Exception:
        d = res.json
    return d.get("res", d)


def read(ocr, path: Path) -> dict:
    """One PaddleOCR reading of a page: boxes, texts, confidences, orientation."""
    res = as_dict(next(iter(ocr.predict(str(path)))))
    items = []
    for text, box, score in zip(res.get("rec_texts", []), res.get("rec_boxes", []), res.get("rec_scores", [])):
        x1, y1, x2, y2 = (float(v) for v in box)
        items.append({"x1": x1, "y1": y1, "x2": x2, "y2": y2,
                      "conf": round(float(score) * 100, 1), "text": str(text)})
    angle = int((res.get("doc_preprocessor_res") or {}).get("angle", 0) or 0)
    conf = round(sum(r["conf"] for r in items) / len(items), 1) if items else 0.0
    return {"items": items, "angle": max(angle, 0), "conf": conf}


def is_footer(r: dict, limit: float) -> bool:
    if any(m.lower() in r["text"].lower() for m in FOOTER_MARKERS):
        return True
    return (r["y1"] + r["y2"]) / 2 > limit and bool(FOOTER_WORDS.search(r["text"]))


def drop_footer(items: list[dict], page_height: float) -> list[dict]:
    """Remove the archive's footer: its exact words anywhere, or footer-like text
    (even misread, e.g. "Sou..ucs") in the bottom band of the page."""
    limit = page_height * (1 - FOOTER_BAND)
    return [r for r in items if not is_footer(r, limit)]


def page_height(path: Path, rotated: int) -> float:
    """Height of the page as PaddleOCR read it (after turning and any shrinking)."""
    with Image.open(path) as img:
        width, height = img.size
    if rotated in (90, 270):
        width, height = height, width
    return height * min(1.0, MAX_SIDE / max(width, height))


def to_lines(items: list[dict]) -> list[str]:
    """Group text boxes into reading lines: top to bottom, then left to right."""
    rows = sorted(items, key=lambda r: (r["y1"] + r["y2"]) / 2)
    lines, current, cy, ch = [], [], None, None
    for r in rows:
        yc, h = (r["y1"] + r["y2"]) / 2, max(r["y2"] - r["y1"], 1)
        if current and abs(yc - cy) > 0.5 * max(h, ch):
            lines.append(current)
            current = []
        if not current:
            cy, ch = yc, h
        current.append(r)
    if current:
        lines.append(current)
    return [" ".join(r["text"] for r in sorted(line, key=lambda r: r["x1"])) for line in lines]


def ocr_page(ocr_auto, ocr_plain, doc_id: str) -> dict:
    """Read one page, keep the better orientation and save its text, boxes and results row."""
    start = time.time()
    path = IMAGE_DIR / f"{doc_id}.png"
    auto = read(ocr_auto, path)
    chosen, rotated = auto, auto["angle"]
    if auto["angle"]:
        plain = read(ocr_plain, path)
        if plain["conf"] >= auto["conf"]:
            chosen, rotated = plain, 0

    items = drop_footer(chosen["items"], page_height(path, rotated))
    text = "\n".join(to_lines(items))
    (OUT_DIR / f"{doc_id}.txt").write_text(text, encoding="utf-8")
    # Box coordinates are in the page as read (turned by "rotated" degrees if not 0)
    pd.DataFrame(items, columns=["x1", "y1", "x2", "y2", "conf", "text"]).to_csv(
        BOX_DIR / f"{doc_id}.tsv", sep="\t", index=False)

    confs = [r["conf"] for r in items]
    return {"id": doc_id, "suggested_angle": auto["angle"], "rotated": rotated,
            "words": len(text.split()),
            "mean_conf": round(sum(confs) / len(confs), 1) if confs else 0.0,
            "chars": len(text), "seconds": round(time.time() - start, 2), "error": ""}


def page_order() -> list[str]:
    """Return the page IDs to read: the OCR test pages first, then the rest."""
    ids = pd.read_csv(DATA / "manifest.csv", dtype=str)["id"].tolist()
    first = pd.read_csv(TEST_PAGES, dtype={"id": str})["id"].tolist() if TEST_PAGES.exists() else []
    seen, out = set(), []
    for i in first + ids:
        if i not in seen and i not in NOT_RETRIEVED and (IMAGE_DIR / f"{i}.png").exists():
            seen.add(i)
            out.append(i)
    return out


def refilter() -> None:
    """Re-apply the footer rule to finished pages from their saved boxes (no OCR, no GPU).
    Box files keep every box PaddleOCR returned before footer removal only for pages run
    after this mode was added; for earlier pages the boxes are those kept at the time."""
    res = pd.read_csv(RESULTS, dtype={"id": str}, keep_default_na=False)
    changed = 0
    for i, row in res.iterrows():
        box_file = BOX_DIR / f"{row['id']}.tsv"
        if not box_file.exists():
            continue
        boxes = pd.read_csv(box_file, sep="\t", keep_default_na=False, dtype={"text": str})
        items = boxes.to_dict("records")
        kept = drop_footer(items, page_height(IMAGE_DIR / f"{row['id']}.png", int(row["rotated"] or 0)))
        if len(kept) == len(items):
            continue
        text = "\n".join(to_lines(kept))
        (OUT_DIR / f"{row['id']}.txt").write_text(text, encoding="utf-8")
        pd.DataFrame(kept, columns=["x1", "y1", "x2", "y2", "conf", "text"]).to_csv(box_file, sep="\t", index=False)
        confs = [float(r["conf"]) for r in kept]
        res.loc[i, ["words", "chars", "mean_conf"]] = [len(text.split()), len(text),
                                                     round(sum(confs) / len(confs), 1) if confs else 0.0]
        changed += 1
    res.to_csv(RESULTS, index=False)
    print(f"Footer rule re-applied: {changed} pages changed.")


def main() -> None:
    """Read every page not yet in the results."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--limit", type=int, default=None, help="only do this many new pages")
    parser.add_argument("--refilter", action="store_true",
                        help="re-apply the footer rule to finished pages, without running OCR")
    args = parser.parse_args()
    if args.refilter:
        refilter()
        return

    BOX_DIR.mkdir(parents=True, exist_ok=True)
    done = resume_csv(RESULTS, "id", dtype={"id": str})
    todo = [i for i in page_order() if i not in done]
    if args.limit:
        todo = todo[: args.limit]
    print(f"Already done: {len(done)}. To do now: {len(todo)}.")
    if not todo:
        return

    from paddleocr import PaddleOCR   # imported here so --help works without loading models
    import paddle
    import paddleocr

    t0 = time.time()
    ocr_auto = PaddleOCR(**AUTO)
    ocr_plain = PaddleOCR(**PLAIN)
    load_seconds = round(time.time() - t0, 1)
    print(f"Models loaded in {load_seconds} s")
    RUN_INFO.write_text(json.dumps({
        "settings_auto": AUTO, "settings_plain": PLAIN, "footer_band": FOOTER_BAND,
        "paddleocr": paddleocr.__version__, "paddle": paddle.__version__,
        "model_load_seconds": load_seconds, "started": time.ctime()}, indent=1), encoding="utf-8")

    progress, errors = Progress(len(todo)), 0
    for n, doc_id in enumerate(todo, 1):
        try:
            row = ocr_page(ocr_auto, ocr_plain, doc_id)
            errors = 0
        except KeyboardInterrupt:
            print("Stopped. Finished pages are saved; run again to continue.")
            return
        except Exception as e:
            row = {"id": doc_id, "suggested_angle": 0, "rotated": 0, "words": 0, "mean_conf": 0.0,
                   "chars": 0, "seconds": 0.0, "error": str(e)[:200]}
            errors += 1
        pd.DataFrame([row]).to_csv(RESULTS, mode="a", header=not RESULTS.exists(), index=False)
        if errors >= MAX_ERRORS_IN_A_ROW:
            print(f"Stopped: {MAX_ERRORS_IN_A_ROW} pages failed in a row. Last error: {row['error']}")
            return
        progress(n)
    print(f"Finished. Results: {RESULTS}")


if __name__ == "__main__":
    main()
