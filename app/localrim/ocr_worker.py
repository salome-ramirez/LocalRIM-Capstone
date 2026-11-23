"""PaddleOCR helper for the intake pipeline.

The app runs in .venv (Python 3.14), which has no PaddleOCR; PaddleOCR is installed in
.venv-ocr (Python 3.12). The pipeline starts this module with .venv-ocr's Python and keeps
it running, so the models load once. It uses run_ocr_paddle's reading rules unchanged:
default models, page read both ways, text-line orientation on, footer removed.

Protocol, one JSON line per page:
    in   {"path": "C:/.../page.png"}
    out  @@RESULT {"text": ..., "items": [...], "rotated": 0, "mean_conf": 97.1,
                   "words": 212, "page_height": ..., "seconds": 0.9, "error": ""}
Lines without the @@RESULT prefix are PaddleOCR's own log output and are ignored.

Usage (from the repository root):
    .venv-ocr\\Scripts\\python -m app.localrim.ocr_worker page.png   # one page, prints the result
    .venv-ocr\\Scripts\\python -m app.localrim.ocr_worker --serve    # what the pipeline uses
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import pipeline.common  # noqa: F401  (puts pipeline/02_ocr on the import path)
from run_ocr_paddle import AUTO, PLAIN, drop_footer, page_height, read, to_lines

PREFIX = "@@RESULT "


def ocr(ocr_auto, ocr_plain, path: Path) -> dict:
    """Read one page with the same steps as run_ocr_paddle.ocr_page and return the result."""
    start = time.time()
    auto = read(ocr_auto, path)
    chosen, rotated = auto, auto["angle"]
    if auto["angle"]:
        plain = read(ocr_plain, path)
        if plain["conf"] >= auto["conf"]:
            chosen, rotated = plain, 0
    height = page_height(path, rotated)
    items = drop_footer(chosen["items"], height)
    text = "\n".join(to_lines(items))
    confs = [r["conf"] for r in items]
    return {"text": text, "items": items, "rotated": rotated, "suggested_angle": auto["angle"],
            "mean_conf": round(sum(confs) / len(confs), 1) if confs else 0.0,
            "words": len(text.split()), "page_height": height,
            "seconds": round(time.time() - start, 2), "error": ""}


def main() -> None:
    """Read the page given on the command line, or serve pages from stdin with --serve."""
    from paddleocr import PaddleOCR
    ocr_auto, ocr_plain = PaddleOCR(**AUTO), PaddleOCR(**PLAIN)

    def answer(path: str) -> None:
        try:
            res = ocr(ocr_auto, ocr_plain, Path(path))
        except Exception as e:
            res = {"error": f"{type(e).__name__}: {str(e)[:300]}"}
        print(PREFIX + json.dumps(res), flush=True)

    if len(sys.argv) > 1 and sys.argv[1] != "--serve":
        answer(sys.argv[1])
        return
    print(PREFIX + json.dumps({"ready": True}), flush=True)
    for line in sys.stdin:
        line = line.strip()
        if line:
            answer(json.loads(line)["path"])


if __name__ == "__main__":
    main()
