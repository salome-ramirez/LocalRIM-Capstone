"""First OCR spot-check app: rate engine A's reading of 10 pages.

Shows one page at a time like the labeling app: the page image on the left, the OCR text
and the rating choices on the right. Ratings are saved as they are made; reopening
continues at the first unrated page.

The 10 pages (seed 44, picked once and saved): 2 Quality and safety pages (one under 60
confidence, one 60-85), 1 German page, and 7 pages spread across confidence bands (under
40: 1, 40-60: 2, 60-75: 1, 75-85: 1, 85 and up: 2). extend_test_pages.py later extends
this list to cover every scan-quality group.

Reads data/pm_sample/manifest.csv, data/pm_sample/ocr/ (text and ocr_results.csv) and
labels/label_pm_labels_<annotator>.csv.
Writes results/ocr_spotcheck_pm_pages.csv (the pages) and results/ocr_spotcheck_pm.csv
(ratings; the newest save per page counts).

Usage:
    streamlit run pipeline/02_ocr/ocr_spotcheck_pm.py
"""

from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st
from PIL import Image

# Settings

REPO = Path(__file__).resolve().parents[2]
PM_DIR = REPO / "data" / "pm_sample"
OCR_DIR = PM_DIR / "ocr"
LABEL_FILE = REPO / "labels" / "label_pm_labels_salo.csv"
OUT_DIR = REPO / "results"
PAGES_FILE = OUT_DIR / "ocr_spotcheck_pm_pages.csv"
OUT_FILE = OUT_DIR / "ocr_spotcheck_pm.csv"

SEED = 44
BANDS = [  # (label, low, high, how many)
    ("under 40", 0, 40, 1),
    ("40-60", 40, 60, 2),
    ("60-75", 60, 75, 1),
    ("75-85", 75, 85, 1),
    ("85 and up", 85, 101, 2),
]

# Rating tiles, with the caption shown under the pick (same idea as the
# department question in the labeler).
RATINGS = {
    "Good": "Text reads correctly; a model could use it as is.",
    "Usable": "Main content is there; some words, numbers or lines are wrong or missing.",
    "Poor": "Too much is missing or garbled to tell what the page is.",
}
ISSUES = [
    "Sideways Bates / stamps",
    "Tables or columns",
    "Handwriting",
    "Underlines dropped lines",
    "Numbers misread",
    "Headers or logos lost",
    "Faint or low contrast",
    "Rotated",
    "German or other language",
]

OUT_DIR.mkdir(exist_ok=True)


# Picking the 10 pages (once)

def find_col(df: pd.DataFrame, *names, contains=None) -> str:
    """Return the first column whose name matches one of the candidates, or contains one."""
    for n in names:
        if n in df.columns:
            return n
    if contains:
        for c in df.columns:
            if contains in c.lower():
                return c
    raise KeyError(f"No column like {names or contains} in {list(df.columns)}")


def build_pool() -> pd.DataFrame:
    """Return every read page with its confidence, image, document number and, for labeled pages, department."""
    # ocr_results.csv columns: path, rotated, suggested, words, mean_conf,
    # chars, seconds, error. The document id is the image file name (path stem).
    ocr = pd.read_csv(OCR_DIR / "ocr_results.csv", dtype={"path": str})
    ocr["id"] = ocr["path"].map(lambda p: Path(str(p).replace("\\", "/")).stem)
    ocr = ocr.rename(columns={"mean_conf": "ocr_conf"})
    if "error" in ocr.columns:   # keep only pages that read without an error
        ocr = ocr[ocr["error"].isna() | (ocr["error"].astype(str).str.strip() == "")]

    manifest = pd.read_csv(PM_DIR / "manifest.csv", dtype=str)
    manifest["position"] = range(1, len(manifest) + 1)   # document number = manifest order
    pool = ocr.merge(manifest[["id", "image", "position"]], on="id", how="left")

    labels = pd.read_csv(LABEL_FILE, dtype=str, keep_default_na=False)
    labels = labels.drop_duplicates(subset="path", keep="last")   # newest save wins
    labels = labels[labels["outcome"] == "labeled"][["doc_id", "function_id", "note"]]
    pool = pool.merge(labels, left_on="id", right_on="doc_id", how="left")
    pool["ocr_conf"] = pd.to_numeric(pool["ocr_conf"], errors="coerce")
    return pool.dropna(subset=["ocr_conf"])


def get_pages() -> pd.DataFrame:
    # Reuse the saved list, so the 10 pages never change between sessions.
    # Delete the file only if you want a new pick.
    """Return the 10 spot-check pages, picking and saving them on the first run."""
    if PAGES_FILE.exists():
        return pd.read_csv(PAGES_FILE, dtype={"id": str})

    pool = build_pool()
    picked = []

    def take(part: pd.DataFrame, n: int, why: str):
        part = part[~part["id"].isin([p[0] for p in picked])]
        for _, r in part.sample(min(n, len(part)), random_state=SEED).iterrows():
            picked.append((r["id"], why))

    qsh = pool[pool["function_id"] == "QSH"]
    take(qsh[qsh["ocr_conf"] < 60], 1, "Quality & Safety, under 60")
    take(qsh[(qsh["ocr_conf"] >= 60) & (qsh["ocr_conf"] < 85)], 1, "Quality & Safety, 60-85")
    take(pool[pool["note"].fillna("").str.contains("German", case=False)], 1, "German page")
    for label, lo, hi, n in BANDS:
        take(pool[(pool["ocr_conf"] >= lo) & (pool["ocr_conf"] < hi)], n, f"Confidence {label}")

    pages = pd.DataFrame(picked, columns=["id", "why_picked"]).merge(pool, on="id", how="left")
    cols = ["id", "position", "why_picked", "ocr_conf", "function_id", "image"]
    if "words" in pages.columns:
        cols.insert(4, "words")
    pages = pages[cols].sort_values("ocr_conf").reset_index(drop=True)
    pages.to_csv(PAGES_FILE, index=False)
    return pages


def read_saved() -> pd.DataFrame:
    """Read the saved ratings, keeping each page's newest one."""
    if not OUT_FILE.exists():
        return pd.DataFrame()
    saved = pd.read_csv(OUT_FILE, dtype=str, keep_default_na=False)
    return saved.drop_duplicates(subset="id", keep="last")   # newest save wins


def save_row(row: dict):
    """Append one rating."""
    pd.DataFrame([row]).to_csv(OUT_FILE, mode="a", header=not OUT_FILE.exists(), index=False)


def image_path(page) -> Path:
    """Return the image path of a page."""
    p = Path(str(page["image"]))
    for cand in [p, PM_DIR / p, PM_DIR / "images" / p.name, PM_DIR / "images" / f"{page['id']}.png"]:
        if cand.exists():
            return cand
    return None


# Page layout (same CSS as labeler.py)

st.set_page_config(page_title="LocalRIM · OCR spot-check", layout="wide")

fit_to_screen = st.sidebar.checkbox(
    "Fit page to window", value=True,
    help="Clear to view the page at full width.")

style = """
[data-testid="stElementToolbar"], [data-testid="StyledFullScreenButton"] {
    display: none !important;
}
.st-key-primary_box {
    background: #FFFFFF;
    border: 2px solid #FF4B4B;
    border-radius: 10px;
    padding: 10px 14px;
}
.st-key-alsofits_box {
    background: #F4F4F5;
    border: 1px solid #E2E2E5;
    border-radius: 10px;
    padding: 8px 12px;
}
.st-key-alsofits_box p { color: #6B6B73; }
.st-key-alsofits_box button[data-variant="pills"] {
    background: #FFFFFF !important;
}
.st-key-alsofits_box button[data-variant="pills"][aria-checked="true"],
.st-key-alsofits_box button[data-variant="pills"][data-selected="true"] {
    background: #E4E4E7 !important;
    border-color: #8A8A93 !important;
    color: #2E2E33 !important;
}
div[data-testid="stButtonGroup"], div[data-testid="stButtonGroup"] > div,
div[data-testid="stButtonGroup"] div[role="group"],
div[data-testid="stButtonGroup"] div[role="radiogroup"] {
    flex-wrap: wrap !important;
    overflow: visible !important;
    white-space: normal !important;
}
"""
if fit_to_screen:
    style += """
div[data-testid="stHorizontalBlock"] { align-items: flex-start; }
div[data-testid="stHorizontalBlock"] > div:first-child {
    position: sticky;
    top: 3.5rem;
}
div[data-testid="stHorizontalBlock"] > div:first-child img {
    max-height: calc(100vh - 9rem);
    object-fit: contain;
}
"""
st.markdown(f"<style>{style}</style>", unsafe_allow_html=True)

pages = get_pages()
queue = pages["id"].tolist()
saved = read_saved()
done = set(saved["id"]) if not saved.empty else set()

# --- Sidebar: progress and navigation (as in the labeler) ---
done_count = sum(p in done for p in queue)
st.sidebar.write(f"**{done_count} / {len(queue)}** pages rated")
st.sidebar.progress(done_count / len(queue))

st.sidebar.caption("Set to 0 to resume at the next unrated page, "
                   "or enter a position to review a saved rating.")
if st.session_state.pop("_reset_jump", False) or "jump_to" not in st.session_state:
    st.session_state["jump_to"] = 0
jump_to = st.sidebar.number_input("Go to position", min_value=0,
                                  max_value=len(queue), step=1, key="jump_to")
st.sidebar.button("Back to where I left off", disabled=jump_to == 0,
                  on_click=lambda: st.session_state.update(jump_to=0))

with st.sidebar.expander("How to rate", expanded=False):
    for name, meaning in RATINGS.items():
        st.markdown(f"**{name}**: {meaning}")
    st.caption("Judge whether the text would let you (or a model) tell what the "
               "page is. Small misreads on a long page can still be Good.")

if jump_to > 0:
    current_index = jump_to - 1
else:
    remaining = [i for i, p in enumerate(queue) if p not in done]
    current_index = remaining[0] if remaining else None
    if current_index is None:
        st.success("All 10 pages rated.")

# --- Main area: image on the left, OCR text and choices on the right ---
if current_index is not None:
    page = pages.iloc[current_index]
    k = page["id"]
    left, right = st.columns([3, 2])

    with left:
        st.subheader(f"Page {current_index + 1} of {len(queue)} · document {page['position']}")
        img = image_path(page)
        if img:
            st.image(Image.open(img).convert("RGB"), width="stretch")
        else:
            st.warning("Image not found.")

    with right:
        prev = {}
        if k in done:
            prev = saved.loc[saved["id"] == k].iloc[-1].to_dict()
            st.info(f"Already saved ({prev.get('saved_at', '')}). Change it and press "
                    "Save to replace it, or use 'Back to where I left off'.")

        words = f" · {int(page['words'])} words" if "words" in page and pd.notna(page.get("words")) else ""
        st.caption(f"{page['why_picked']} · OCR confidence {page['ocr_conf']:.1f}{words}")
        txt = OCR_DIR / f"{k}.txt"
        text = txt.read_text(encoding="utf-8", errors="replace") if txt.exists() else "(no OCR text file)"
        st.text_area("OCR text", text, height=320, key=f"text_{k}")

        with st.container(key="primary_box"):
            st.caption("How well did it read?")
            rating = st.pills("Rating", list(RATINGS), selection_mode="single",
                              default=prev.get("rating") or None,
                              label_visibility="collapsed", key=f"rating_{k}")
            if rating:
                st.caption(RATINGS[rating])

        st.divider()
        prev_issues = [i for i in prev.get("issues", "").split("; ") if i in ISSUES]
        with st.container(key="alsofits_box"):
            st.markdown("**What went wrong** (optional)")
            issues = st.pills("Issues", ISSUES, selection_mode="multi",
                              default=prev_issues,
                              label_visibility="collapsed", key=f"issues_{k}")

        note = st.text_input("Note", value=prev.get("note", ""),
                             placeholder="Required for Poor. What was lost or misread?",
                             key=f"note_{k}")

        if st.button("Save and continue", type="primary", key=f"save_{k}"):
            if rating is None:
                st.warning("Select a rating.")
            elif rating == "Poor" and not note.strip():
                st.warning("A note is required for Poor pages.")
            else:
                save_row({
                    "id": k,
                    "position": page["position"],
                    "why_picked": page["why_picked"],
                    "ocr_conf": page["ocr_conf"],
                    "rating": rating,
                    "issues": "; ".join(issues),
                    "note": note.strip(),
                    "saved_at": datetime.now().isoformat(timespec="seconds"),
                })
                if jump_to > 0:   # a revisited page: go back to the next unrated one
                    st.session_state["_reset_jump"] = True
                st.rerun()

# Summary

if not saved.empty:
    st.divider()
    with st.expander("Spot-check summary", expanded=current_index is None):
        rated = pages.merge(saved[["id", "rating", "issues", "note"]], on="id", how="inner")
        st.write(", ".join(f"{(rated['rating'] == r).sum()} {r.lower()}" for r in RATINGS))
        st.dataframe(rated[["position", "why_picked", "ocr_conf", "rating", "issues", "note"]],
                     hide_index=True)
