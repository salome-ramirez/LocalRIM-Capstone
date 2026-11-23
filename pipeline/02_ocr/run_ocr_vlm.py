"""OCR engine D: a local vision-language OCR model through Ollama.

Sends each first page (scan resolution, longest side shrunk to MAX_SIDE pixels) to an
OCR vision model served by Ollama on this computer and saves the transcription and the
time per page, in the same layout as the other engines. These models give no word
positions or confidence, so engine D is compared as a reader but can't be the main
engine.

Settings: MODEL (default "glm-ocr", `ollama pull glm-ocr`), PROMPT, MAX_SIDE; lines with
the archive's source footer are removed.

Writes:
    data/pm_sample/ocr_vlm/<id>.txt          transcription
    data/pm_sample/ocr_vlm/ocr_results.csv   id, words, chars, seconds, error
    data/pm_sample/ocr_vlm/run_info.json     model, prompt, settings

Order: the OCR test pages first, then the rest. Resumable.

Usage:
    python -m pipeline.02_ocr.run_ocr_vlm --limit 3
    python -m pipeline.02_ocr.run_ocr_vlm
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import time
from pathlib import Path

import pandas as pd
import requests
from PIL import Image

from pipeline.common import Progress, ROOT, resume_csv

DATA = ROOT / "data" / "pm_sample"
IMAGE_DIR = DATA / "images_native"
OUT_DIR = DATA / "ocr_vlm"
RESULTS = OUT_DIR / "ocr_results.csv"
RUN_INFO = OUT_DIR / "run_info.json"
TEST_PAGES = ROOT / "results" / "ocr_spotcheck_pm_pages.csv"

OLLAMA = "http://localhost:11434/api/generate"
MODEL = "glm-ocr"
PROMPT = ("Transcribe all the text on this scanned page exactly as written, line by line, "
          "top to bottom. Do not describe the page, summarise, translate or add anything.")
MAX_SIDE = 1600
TIMEOUT = 300                  # seconds per page
NOT_RETRIEVED = {"qnkl0085", "hzgp0030", "xpxk0085"}   # never downloaded
FOOTER_MARKERS = ("industrydocuments", "Source: http")
MAX_ERRORS_IN_A_ROW = 5


def encode(path: Path) -> str:
    """Return the page image, shrunk to MAX_SIDE, as base64 PNG."""
    with Image.open(path) as img:
        img = img.convert("RGB")
        img.thumbnail((MAX_SIDE, MAX_SIDE))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def ocr_page(model: str, doc_id: str) -> dict:
    """Send one page to the model, save its transcription and return its results row."""
    start = time.time()
    r = requests.post(OLLAMA, json={"model": model, "prompt": PROMPT,
                                    "images": [encode(IMAGE_DIR / f"{doc_id}.png")],
                                    "stream": False, "options": {"temperature": 0}},
                      timeout=TIMEOUT)
    r.raise_for_status()
    text = r.json().get("response", "")
    text = "\n".join(l for l in text.splitlines() if not any(m.lower() in l.lower() for m in FOOTER_MARKERS))
    (OUT_DIR / f"{doc_id}.txt").write_text(text, encoding="utf-8")
    return {"id": doc_id, "words": len(text.split()), "chars": len(text),
            "seconds": round(time.time() - start, 2), "error": ""}


def page_order() -> list[str]:
    """Return the page IDs: the OCR test pages first, then the rest."""
    ids = pd.read_csv(DATA / "manifest.csv", dtype=str)["id"].tolist()
    first = pd.read_csv(TEST_PAGES, dtype={"id": str})["id"].tolist() if TEST_PAGES.exists() else []
    seen, out = set(), []
    for i in first + ids:
        if i not in seen and i not in NOT_RETRIEVED and (IMAGE_DIR / f"{i}.png").exists():
            seen.add(i)
            out.append(i)
    return out


def main() -> None:
    """Read every page not yet in the results."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--limit", type=int, default=None, help="only do this many new pages")
    parser.add_argument("--model", default=MODEL, help="Ollama model name")
    args = parser.parse_args()

    try:
        requests.get("http://localhost:11434/api/tags", timeout=5).raise_for_status()
    except Exception:
        raise SystemExit("Ollama is not running. Start the Ollama app (or run `ollama serve`) and try again.")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    done = resume_csv(RESULTS, "id", dtype={"id": str})
    todo = [i for i in page_order() if i not in done]
    if args.limit:
        todo = todo[: args.limit]
    print(f"Already done: {len(done)}. To do now: {len(todo)}.")
    RUN_INFO.write_text(json.dumps({"model": args.model, "prompt": PROMPT, "max_side": MAX_SIDE,
                                    "temperature": 0, "started": time.ctime()}, indent=1), encoding="utf-8")

    progress, errors = Progress(len(todo), every=5), 0
    for n, doc_id in enumerate(todo, 1):
        try:
            row = ocr_page(args.model, doc_id)
            errors = 0
        except KeyboardInterrupt:
            print("Stopped. Finished pages are saved; run again to continue.")
            return
        except Exception as e:
            row = {"id": doc_id, "words": 0, "chars": 0, "seconds": 0.0, "error": str(e)[:200]}
            errors += 1
        pd.DataFrame([row]).to_csv(RESULTS, mode="a", header=not RESULTS.exists(), index=False)
        if errors >= MAX_ERRORS_IN_A_ROW:
            print(f"Stopped: {MAX_ERRORS_IN_A_ROW} pages failed in a row. Last error: {row['error']}")
            return
        progress(n)
    print(f"Finished. Results: {RESULTS}")


if __name__ == "__main__":
    main()
