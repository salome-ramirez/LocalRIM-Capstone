"""OCR accuracy on FUNSD's 50 test forms, for engines A, C and D.

FUNSD gives word-level ground truth, so each engine's reading can be checked against the
true text instead of rated by eye. Word recall is the first criterion of the engine
choice.

Engines (settings imported from the scripts that read the sample pages):
    A  Tesseract, settings of run_ocr.py (2x upscale, --psm 3)
    C  PaddleOCR, settings of run_ocr_paddle.py
    D  GLM-OCR through Ollama, settings of run_ocr_vlm.py
    (E, the archive's text layer, doesn't exist for FUNSD forms.)

Stages: "ocr" reads every form and saves its text and timing; "score" compares the text
with FUNSD's words; "both" runs both. PaddleOCR is in .venv-ocr, so for C run the ocr
stage there and score in .venv.

Metrics: WER and CER (lower is better); word_recall, the share of true words found
anywhere on the page, order ignored; mean_conf, the engine's own confidence (scales
differ between engines).

Writes:
    results/funsd_ocr_eval.csv, results/funsd_ocr_text/            engine A
    results/funsd_ocr_eval_<engine>.csv                            one row per form
    results/funsd_ocr_text_<engine>/                               the text of each form
    results/funsd_ocr_raw_<engine>.csv                             timing and confidence

Usage:
    python pipeline/02_ocr/eval_ocr_funsd.py                      # engine A
    python pipeline/02_ocr/eval_ocr_funsd.py --engine D           # Ollama running
    .venv-ocr\\Scripts\\python pipeline/02_ocr/eval_ocr_funsd.py --engine C --stage ocr
    python pipeline/02_ocr/eval_ocr_funsd.py --engine C --stage score
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import time
from collections import Counter
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
FUNSD_DIR = ROOT / "data" / "funsd"
OUT_DIR = ROOT / "results"

# Engine settings. C and D import theirs from the scripts that read the 1,400
# Philip Morris pages, so FUNSD measures exactly the same engines.

# A: run_ocr.py
TESSERACT = Path(r"C:\Program Files\Tesseract-OCR\tesseract.exe")
UPSCALE = 2
TESSERACT_CONFIG = "--psm 3"

# C: run_ocr_paddle.py  (models, AUTO/PLAIN settings, read both ways, line grouping)
# D: run_ocr_vlm.py     (model, prompt, image shrinking, temperature 0)
# The archive footer filters in those scripts are not needed: FUNSD forms have no footer.


# Reading the FUNSD files

def find_image(json_path: Path, ann: dict) -> Path | None:
    """Return a form's image: next to its JSON, or at the path stored inside."""
    for ext in (".png", ".jpg", ".jpeg"):
        candidate = json_path.with_suffix(ext)
        if candidate.exists():
            return candidate
    for key in ("image_path", "image", "file_name"):
        if isinstance(ann.get(key), str):
            candidate = json_path.parent / Path(ann[key]).name
            if candidate.exists():
                return candidate
    return None


def get_words_and_boxes(ann: dict, json_path: Path):
    """Return a form's true words and their boxes."""
    words = ann.get("words") or ann.get("tokens")
    boxes = ann.get("boxes") or ann.get("bboxes")
    if words is None or boxes is None:
        raise KeyError(f"{json_path.name}: expected 'words'/'tokens' and 'boxes'/'bboxes', "
                       f"found keys {list(ann.keys())}")
    return words, boxes


def reading_order(items: list[tuple[str, list[float]]]) -> str:
    """Return words with boxes (x0, y0, x1, y1) as lines: top to bottom, then left to right."""
    items = [(str(w), b) for w, b in items if str(w).strip()]
    if not items:
        return ""
    heights = sorted(b[3] - b[1] for _, b in items)
    half_line = max(heights[len(heights) // 2], 1) / 2

    items.sort(key=lambda t: (t[1][1] + t[1][3]) / 2)
    lines = []
    for word, box in items:
        y_center = (box[1] + box[3]) / 2
        if lines and abs(y_center - lines[-1]["y"]) <= half_line:
            lines[-1]["words"].append((box[0], word))
        else:
            lines.append({"y": y_center, "words": [(box[0], word)]})
    return "\n".join(" ".join(w for _, w in sorted(line["words"])) for line in lines)


def form_files(split: str) -> list[Path]:
    """Return the JSON files of the forms in a split."""
    folders = ["train", "test"] if split == "all" else [split]
    return sorted(p for f in folders for p in (FUNSD_DIR / f).glob("*.json"))


# Engines: each returns (text, mean confidence or None)

def make_tesseract():
    """Return engine A's reader."""
    import pytesseract
    if TESSERACT.exists():
        pytesseract.pytesseract.tesseract_cmd = str(TESSERACT)

    def read(image_path: Path):
        img = Image.open(image_path).convert("RGB")
        img = img.resize((img.width * UPSCALE, img.height * UPSCALE), Image.LANCZOS)
        data = pytesseract.image_to_data(img, config=TESSERACT_CONFIG,
                                         output_type=pytesseract.Output.DICT)
        lines, confs = {}, []
        for i, raw in enumerate(data["text"]):
            word = str(raw).strip()
            conf = float(data["conf"][i])
            if not word or conf < 0:
                continue
            key = (data["block_num"][i], data["par_num"][i], data["line_num"][i])
            lines.setdefault(key, []).append(word)
            confs.append(conf)
        text = "\n".join(" ".join(ws) for ws in lines.values())
        return text, (sum(confs) / len(confs) if confs else 0.0)
    return read


def make_paddle():
    """Return engine C's reader, with run_ocr_paddle.py's settings.

    Reads with page orientation on; if the model suggests turning the page, also reads it
    as scanned and keeps the more confident reading.
    """
    import run_ocr_paddle as paddle_run
    from paddleocr import PaddleOCR
    ocr_auto = PaddleOCR(**paddle_run.AUTO)
    ocr_plain = PaddleOCR(**paddle_run.PLAIN)

    def read(image_path: Path):
        auto = paddle_run.read(ocr_auto, image_path)
        chosen = auto
        if auto["angle"]:
            plain = paddle_run.read(ocr_plain, image_path)
            if plain["conf"] >= auto["conf"]:
                chosen = plain
        return "\n".join(paddle_run.to_lines(chosen["items"])), chosen["conf"]
    return read


def make_vlm():
    """Return engine D's reader: run_ocr_vlm.py's model, prompt, image size and temperature 0."""
    import requests
    import run_ocr_vlm as vlm_run
    try:
        requests.get("http://localhost:11434/api/tags", timeout=5).raise_for_status()
    except Exception:
        raise SystemExit("Ollama is not running. Start the Ollama app and try again.")

    def read(image_path: Path):
        r = requests.post(vlm_run.OLLAMA, json={
            "model": vlm_run.MODEL, "prompt": vlm_run.PROMPT,
            "images": [vlm_run.encode(image_path)],
            "stream": False, "options": {"temperature": 0}}, timeout=vlm_run.TIMEOUT)
        r.raise_for_status()
        return r.json().get("response", ""), None
    return read


ENGINES = {"A": make_tesseract, "C": make_paddle, "D": make_vlm}


def paths_for(engine: str) -> dict[str, Path]:
    """Return an engine's output paths (A keeps its original names)."""
    suffix = "" if engine == "A" else f"_{engine}"
    return {
        "text": OUT_DIR / f"funsd_ocr_text{suffix}",
        "raw": OUT_DIR / f"funsd_ocr_raw_{engine}.csv",
        "eval": OUT_DIR / f"funsd_ocr_eval{suffix}.csv",
    }


# Stage 1: OCR

def run_ocr(engine: str, split: str) -> None:
    """Read every form with one engine and save its text and timing."""
    paths = paths_for(engine)
    paths["text"].mkdir(parents=True, exist_ok=True)
    files = form_files(split)
    print(f"Engine {engine}: reading {len(files)} FUNSD forms ({split})")

    read = ENGINES[engine]()
    rows = []
    for n, json_path in enumerate(files, 1):
        ann = json.loads(json_path.read_text(encoding="utf-8"))
        image = find_image(json_path, ann)
        if image is None:
            print(f"  [{n}] no image for {json_path.name}, skipped")
            continue
        start = time.time()
        try:
            text, conf = read(image)
            error = ""
        except Exception as exc:  # a failure is recorded, not retried (same rule as the full runs)
            text, conf, error = "", None, f"{type(exc).__name__}: {exc}"
        seconds = round(time.time() - start, 2)
        (paths["text"] / f"{json_path.stem}.txt").write_text(text, encoding="utf-8")
        rows.append({"form": json_path.stem, "split": json_path.parent.name,
                     "mean_conf": "" if conf is None else round(conf, 1),
                     "seconds": seconds, "error": error})
        print(f"  [{n}/{len(files)}] {json_path.stem}: {len(text.split())} words, {seconds}s"
              + (f"  ERROR {error}" if error else ""))

    with paths["raw"].open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["form", "split", "mean_conf", "seconds", "error"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved text to {paths['text']} and timing to {paths['raw']}")


# Stage 2: scoring

def normalize(text: str) -> str:
    """Lowercase, turn punctuation into spaces and collapse spaces (applied to both sides)."""
    text = re.sub(r"[^\w\s]", " ", text.lower())
    return " ".join(text.split())


def word_recall(ref: str, hyp: str) -> float:
    """Return the share of true words found on the page, order ignored, repeats counted."""
    ref_counts, hyp_counts = Counter(ref.split()), Counter(hyp.split())
    return sum((ref_counts & hyp_counts).values()) / max(sum(ref_counts.values()), 1)


def run_score(engine: str, split: str) -> None:
    """Score an engine's saved text against FUNSD and save one row per form."""
    import jiwer
    import pandas as pd

    paths = paths_for(engine)
    if not paths["raw"].exists():
        raise SystemExit(f"No OCR output for engine {engine}: run --stage ocr first.")
    raw = pd.read_csv(paths["raw"], dtype={"form": str}, keep_default_na=False).set_index("form")

    rows, refs, hyps = [], [], []
    for json_path in form_files(split):
        if json_path.stem not in raw.index:
            continue
        ann = json.loads(json_path.read_text(encoding="utf-8"))
        words, boxes = get_words_and_boxes(ann, json_path)
        ref = normalize(reading_order(list(zip(words, boxes))))
        if not ref:
            continue
        text_path = paths["text"] / f"{json_path.stem}.txt"
        hyp = normalize(text_path.read_text(encoding="utf-8")) if text_path.exists() else ""
        info = raw.loc[json_path.stem]
        rows.append({
            "form": json_path.stem,
            "split": json_path.parent.name,
            "true_words": len(ref.split()),
            "ocr_words": len(hyp.split()),
            "wer": round(jiwer.wer(ref, hyp) if hyp else 1.0, 4),
            "cer": round(jiwer.cer(ref, hyp) if hyp else 1.0, 4),
            "word_recall": round(word_recall(ref, hyp), 4),
            "mean_conf": info["mean_conf"],
            "seconds": info["seconds"],
            "error": info["error"],
        })
        refs.append(ref)
        hyps.append(hyp if hyp else "<empty>")

    df = pd.DataFrame(rows)
    df.to_csv(paths["eval"], index=False)

    pooled_recall = (df["word_recall"] * df["true_words"]).sum() / df["true_words"].sum()
    print(f"\n=== FUNSD OCR accuracy, engine {engine} ===")
    print(f"Forms evaluated:        {len(df)}  (failed: {(df['error'] != '').sum()})")
    print(f"True words:             {df['true_words'].sum()}")
    print(f"WER  (pooled):          {jiwer.wer(refs, hyps):.3f}")
    print(f"CER  (pooled):          {jiwer.cer(refs, hyps):.3f}")
    print(f"Word recall (pooled):   {pooled_recall:.3f}")
    print(f"WER per form: median {df['wer'].median():.3f}, "
          f"range {df['wer'].min():.3f} to {df['wer'].max():.3f}")
    print(f"Seconds per form:       {pd.to_numeric(df['seconds']).mean():.2f}")
    conf = pd.to_numeric(df["mean_conf"], errors="coerce")
    if conf.notna().any():
        print(f"Mean confidence:        {conf.mean():.1f} (engine's own scale)")
        # Spearman by hand: correlate the ranks (no scipy needed)
        print(f"  conf vs WER     Spearman {conf.rank().corr(df['wer'].rank()):.2f}")
        print(f"  conf vs recall  Spearman {conf.rank().corr(df['word_recall'].rank()):.2f}")
    print(f"\nSaved: {paths['eval']}")



def main() -> None:
    """Run the chosen stage for the chosen engine."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--engine", choices=sorted(ENGINES), default="A")
    parser.add_argument("--stage", choices=["ocr", "score", "both"], default="both")
    parser.add_argument("--split", choices=["test", "train", "all"], default="test")
    args = parser.parse_args()

    OUT_DIR.mkdir(exist_ok=True)
    if args.stage in ("ocr", "both"):
        run_ocr(args.engine, args.split)
    if args.stage in ("score", "both"):
        run_score(args.engine, args.split)


if __name__ == "__main__":
    main()
