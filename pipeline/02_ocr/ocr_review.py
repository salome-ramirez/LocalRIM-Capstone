"""Blind OCR review app: rate every engine's reading of the test pages side by side.

For each test page the scan is shown beside each engine's text, labelled "Reading 1, 2,
3..." in a shuffled order fixed per page, so the reviewer doesn't know which engine is
which. Each reading is rated on overall quality, words found (of the words on the page,
how many it identified) and spelling (of those, how many are exactly right), with
optional issue tags and a note. The readings on a page are then ranked (no ties). The
ranking supports the ratings but is not part of the winning rule. Engine E, the archive's
own text layer, is a reference, not a candidate.

Below the readings the reviewer confirms the page's scan-quality group (correct, wrong
with the right group, or unsure); the answers are recorded, the groups are not changed.

Reads results/ocr_spotcheck_pm_pages.csv, data/pm_sample/images_native/<id>.png, each
engine's text and data/pm_sample/page_profile.csv.
Writes results/ocr_review_ratings.csv (one row per page and engine, with rank) and
results/scan_group_check.csv (one row per page); the newest save counts.

Usage:
    streamlit run pipeline/02_ocr/ocr_review.py
"""

from __future__ import annotations

import csv
import random
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "pm_sample"
PAGES_FILE = ROOT / "results" / "ocr_spotcheck_pm_pages.csv"
OUT_FILE = ROOT / "results" / "ocr_review_ratings.csv"
PROFILE_FILE = DATA / "page_profile.csv"
GROUP_OUT_FILE = ROOT / "results" / "scan_group_check.csv"
IMAGE_DIRS = [DATA / "images_native", DATA / "images"]
SEED = 44

ENGINE_DIRS = {
    "A": DATA / "ocr",
    "B": DATA / "ocr_tess_native",
    "C": DATA / "ocr_paddle",
    "D": DATA / "ocr_vlm",
    "E": DATA / "ocr_ucsf",
}

RATINGS = {
    "Good": "Text reads correctly; a model could use it as is.",
    "Usable": "Main content is there; some words, numbers or lines are wrong or missing.",
    "Poor": "Too much is missing or garbled to tell what the page is.",
}
SCALE = ["All or nearly all", "Most", "About half", "Few", "None"]
# Two word-level questions:
#   Words found: of the words on the page, how many did the reading identify (spelling aside)?
#   Spelling:    of the words it identified, how many are spelled exactly right?
WORD_QUESTIONS = {
    "words_found": ("Words found", "Of the words on the page, how many did it identify, even if misspelled?"),
    "spelling": ("Spelling", "Of the words it identified, how many are spelled exactly right?"),
}
ISSUES = [
    "Sideways Bates / stamps", "Tables or columns", "Handwriting",
    "Underlines dropped lines", "Numbers misread", "Headers or logos lost",
    "Faint or low contrast", "Scan noise or smudging", "Rotated",
    "German or other language", "Extra junk text",
]

# Scan-quality groups
GROUP_NAMES = {
    "1": "Clean typed files",
    "2": "Photographic ads",
    "3a": "Degraded scans",
    "3b": "Graphics and clippings",
    "4": "Clean reports, forms, tables",
}
GROUP_COLUMNS = ["group", "group_id", "scan_group", "page_group", "group_name"]
GROUP_ANSWERS = ["Correct", "Wrong", "Unsure"]

REFERENCE_ENGINES = {"E"}  # shown and ranked blind, but not a candidate for the winner


# Data

RATING_COLUMNS = ["id", "position", "engine", "reading_slot", "rating", "words_found", "spelling",
                  "issues", "note", "rank", "n_readings", "saved_at"]


def read_all_saved() -> pd.DataFrame:
    """Return every saved rating row.

    Rows written before the rank columns existed (10 fields) and after (12 fields) may
    share one file, so each row is read by its field count.
    """
    if not OUT_FILE.exists():
        return pd.DataFrame(columns=RATING_COLUMNS)
    with OUT_FILE.open(newline="", encoding="utf-8") as f:
        rows = list(csv.reader(f))
    if not rows:
        return pd.DataFrame(columns=RATING_COLUMNS)
    header, records = rows[0], []
    for row in rows[1:]:
        if not row:
            continue
        cols = header if len(row) == len(header) else RATING_COLUMNS if len(row) == len(RATING_COLUMNS) else None
        if cols is None:
            raise ValueError(f"Unexpected row length {len(row)} in {OUT_FILE.name}: {row[:3]}")
        records.append(dict(zip(cols, row)))
    return pd.DataFrame(records).reindex(columns=RATING_COLUMNS, fill_value="").fillna("")


def read_saved() -> pd.DataFrame:
    """Return the saved ratings, keeping the newest save per page and reading."""
    return read_all_saved().drop_duplicates(subset=["id", "engine"], keep="last")


def save_rows(rows: list[dict]) -> None:
    """Append rows, rewriting the file with one header so old and new rows line up."""
    combined = pd.concat([read_all_saved(), pd.DataFrame(rows).astype(str)], ignore_index=True)
    combined.reindex(columns=RATING_COLUMNS, fill_value="").to_csv(OUT_FILE, index=False)


def group_key(value: str) -> str:
    """Return the group code: '3a', '3a Degraded scans' or '3a - Degraded scans' -> '3a'."""
    return str(value).strip().split(" ")[0].strip()


def group_label(key: str) -> str:
    """Return a group's display label."""
    return f"{key} {GROUP_NAMES.get(key, '')}".strip()


def read_groups() -> tuple[dict[str, str], str | None]:
    """Return the scan group per page ID, and an error message if the file can't be used."""
    if not PROFILE_FILE.exists():
        return {}, f"Scan groups not found: {PROFILE_FILE.relative_to(ROOT)}"
    profile = pd.read_csv(PROFILE_FILE, dtype=str, keep_default_na=False)
    col = next((c for c in GROUP_COLUMNS if c in profile.columns), None)
    if "id" not in profile.columns or col is None:
        return {}, f"No id or group column in page_profile.csv (columns: {', '.join(profile.columns)})"
    return {r["id"]: group_key(r[col]) for _, r in profile.iterrows()}, None


def read_group_checks() -> pd.DataFrame:
    """Return the saved group answers, newest per page."""
    if not GROUP_OUT_FILE.exists():
        return pd.DataFrame(columns=["id"])
    checks = pd.read_csv(GROUP_OUT_FILE, dtype=str, keep_default_na=False)
    return checks.drop_duplicates(subset=["id"], keep="last")


def save_group_check(row: dict) -> None:
    """Append one group answer."""
    pd.DataFrame([row]).to_csv(GROUP_OUT_FILE, mode="a", header=not GROUP_OUT_FILE.exists(), index=False)


def page_image(doc_id: str) -> Path | None:
    """Return the image path of a page."""
    for d in IMAGE_DIRS:
        p = d / f"{doc_id}.png"
        if p.exists():
            return p
    return None


def engine_text(engine: str, doc_id: str) -> str | None:
    """Return an engine's text for a page."""
    p = ENGINE_DIRS[engine] / f"{doc_id}.txt"
    return p.read_text(encoding="utf-8", errors="replace") if p.exists() else None


def shuffled(engines: list[str], doc_id: str) -> list[str]:
    """Return the engines in the same blind order every time for a given page."""
    order = sorted(engines)
    random.Random(f"{SEED}-{doc_id}").shuffle(order)
    return order


# Page layout (styles from labeler.py)

st.set_page_config(page_title="LocalRIM · OCR review", layout="wide")
fit_to_screen = st.sidebar.checkbox("Fit page to window", value=True,
                                    help="Clear to view the page at full width.")

style = """
[data-testid="stElementToolbar"], [data-testid="StyledFullScreenButton"] { display: none !important; }
[class*="st-key-primary_box"] {
    background: #FFFFFF; border: 2px solid #FF4B4B; border-radius: 10px; padding: 10px 14px;
}
[class*="st-key-alsofits_box"] {
    background: #F4F4F5; border: 1px solid #E2E2E5; border-radius: 10px; padding: 8px 12px;
}
[class*="st-key-alsofits_box"] p { color: #6B6B73; }
[class*="st-key-alsofits_box"] button[data-variant="pills"] { background: #FFFFFF !important; }
[class*="st-key-alsofits_box"] button[data-variant="pills"][aria-checked="true"],
[class*="st-key-alsofits_box"] button[data-variant="pills"][data-selected="true"] {
    background: #E4E4E7 !important; border-color: #8A8A93 !important; color: #2E2E33 !important;
}
div[data-testid="stButtonGroup"], div[data-testid="stButtonGroup"] > div,
div[data-testid="stButtonGroup"] div[role="group"], div[data-testid="stButtonGroup"] div[role="radiogroup"] {
    flex-wrap: wrap !important; overflow: visible !important; white-space: normal !important;
}
"""
if fit_to_screen:
    style += """
div[data-testid="stHorizontalBlock"] { align-items: flex-start; }
div[data-testid="stHorizontalBlock"] > div:first-child { position: sticky; top: 3.5rem; }
div[data-testid="stHorizontalBlock"] > div:first-child img { max-height: calc(100vh - 9rem); object-fit: contain; }
"""
st.markdown(f"<style>{style}</style>", unsafe_allow_html=True)

if not PAGES_FILE.exists():
    st.error(f"Test pages not found: {PAGES_FILE.relative_to(ROOT)}")
    st.stop()
pages = pd.read_csv(PAGES_FILE, dtype={"id": str})
queue = pages["id"].tolist()

page_groups, group_error = read_groups()
if group_error:
    st.sidebar.warning(group_error)

available = [e for e, d in ENGINE_DIRS.items() if d.exists()]
engines = st.sidebar.multiselect("Engines to review", available, default=available,
                                 help="Engine names are never shown next to the readings.")
if not engines:
    st.info("Select at least one engine in the sidebar.")
    st.stop()

saved = read_saved()
done_pairs = set(zip(saved["id"], saved["engine"]))
group_checks = read_group_checks()
checked_ids = set(group_checks["id"])


def page_done(doc_id: str) -> bool:
    engines_done = all((doc_id, e) in done_pairs or engine_text(e, doc_id) is None for e in engines)
    return engines_done and (doc_id in checked_ids or doc_id not in page_groups)


# --- Sidebar: progress and navigation ---
done_count = sum(page_done(p) for p in queue)
st.sidebar.write(f"**{done_count} / {len(queue)}** pages reviewed")
st.sidebar.progress(done_count / len(queue))
st.sidebar.caption("Set to 0 to resume at the next unreviewed page, or enter a position to review a saved page.")
if st.session_state.pop("_reset_jump", False) or "jump_to" not in st.session_state:
    st.session_state["jump_to"] = 0
jump_to = st.sidebar.number_input("Go to position", min_value=0, max_value=len(queue), step=1, key="jump_to")
st.sidebar.button("Back to where I left off", disabled=jump_to == 0,
                  on_click=lambda: st.session_state.update(jump_to=0))
with st.sidebar.expander("How to rate", expanded=False):
    for name, meaning in RATINGS.items():
        st.markdown(f"**{name}**: {meaning}")
    for label, meaning in WORD_QUESTIONS.values():
        st.markdown(f"**{label}**: {meaning}")
    st.markdown("**Scan group check**: does the page look like the group it was placed in?")

if jump_to > 0:
    current = jump_to - 1
else:
    remaining = [i for i, p in enumerate(queue) if not page_done(p)]
    current = remaining[0] if remaining else None
    if current is None:
        st.success("All pages reviewed for the selected engines.")

# --- Main area: page on the left, blind readings on the right ---
if current is not None:
    doc_id = queue[current]
    page = pages.iloc[current]
    left, right = st.columns([3, 2])

    with left:
        st.subheader(f"Page {current + 1} of {len(queue)} · document {page.get('position', '')}")
        img = page_image(doc_id)
        if img:
            st.image(Image.open(img).convert("RGB"), width="stretch")
        else:
            st.warning("Image not found.")

    with right:
        order = [e for e in shuffled(engines, doc_id) if engine_text(e, doc_id) is not None]
        missing = [e for e in engines if engine_text(e, doc_id) is None]
        if missing:
            st.caption(f"{len(missing)} selected engine(s) have no output for this page.")

        answers = {}
        prevs = {}
        ranks = {}
        n_read = len(order)
        rank_options = list(range(n_read, 0, -1))  # best first
        for slot, engine in enumerate(order, 1):
            prev = {}
            if (doc_id, engine) in done_pairs:
                prev = saved[(saved["id"] == doc_id) & (saved["engine"] == engine)].iloc[-1].to_dict()
            prevs[engine] = prev
            k = f"{doc_id}_{slot}"
            st.markdown(f"#### Reading {slot}")
            st.text_area(f"Reading {slot}", engine_text(engine, doc_id), height=240,
                         label_visibility="collapsed", key=f"text_{k}")
            with st.container(key=f"primary_box_{slot}"):
                st.caption("How well did it read?")
                rating = st.pills("Rating", list(RATINGS), selection_mode="single",
                                  default=prev.get("rating") or None,
                                  label_visibility="collapsed", key=f"rating_{k}")
                words = {}
                for col, (label, meaning) in WORD_QUESTIONS.items():
                    st.caption(f"{label}: {meaning}")
                    words[col] = st.pills(label, SCALE, selection_mode="single",
                                          default=prev.get(col) or None,
                                          label_visibility="collapsed", key=f"{col}_{k}")
            prev_issues = [i for i in prev.get("issues", "").split("; ") if i in ISSUES]
            with st.container(key=f"alsofits_box_{slot}"):
                st.markdown("**What went wrong** (optional)")
                issues = st.pills("Issues", ISSUES, selection_mode="multi", default=prev_issues,
                                  label_visibility="collapsed", key=f"issues_{k}")
            note = st.text_input("Note", value=prev.get("note", ""), key=f"note_{k}",
                                 placeholder="Optional. What was lost or misread?")
            if n_read > 1:
                prev_rank = prev.get("rank", "")
                prev_rank = int(prev_rank) if str(prev_rank).isdigit() and int(prev_rank) in rank_options else None
                with st.container(key=f"primary_box_rank_{slot}"):
                    st.caption(f"Rank against the other readings ({n_read} = best, 1 = worst)")
                    ranks[engine] = st.pills(f"Rank {slot}", rank_options, selection_mode="single",
                                             default=prev_rank, label_visibility="collapsed",
                                             key=f"rank_{doc_id}_{slot}")
            answers[engine] = (slot, rating, words, issues, note)
            st.divider()

        # --- Rank summary: every reading's rank at a glance, repeats flagged ---
        if n_read > 1:
            counts = pd.Series([r for r in ranks.values() if r is not None]).value_counts()
            repeated = set(counts[counts > 1].index)
            lines = []
            for slot, engine in enumerate(order, 1):
                r = ranks.get(engine)
                if r is None:
                    lines.append(f"Reading {slot}: —")
                elif r in repeated:
                    lines.append(f"Reading {slot}: :red[**{r}** (used twice)]")
                else:
                    lines.append(f"Reading {slot}: **{r}**")
            with st.container(key="alsofits_box_ranks"):
                st.markdown("**Your ranks**")
                st.markdown("  \n".join(lines))
                if repeated:
                    st.caption(f"Each number once: {', '.join(map(str, sorted(repeated, reverse=True)))} "
                               f"is used more than once.")
            st.divider()

        # --- Scan group check (one per page, below the readings) ---
        assigned = page_groups.get(doc_id)
        g_answer, g_other, g_note = None, None, ""
        if assigned:
            prev_g = {}
            if doc_id in checked_ids:
                prev_g = group_checks[group_checks["id"] == doc_id].iloc[-1].to_dict()
            with st.container(key="alsofits_box_group"):
                st.markdown("**Scan group check**")
                st.caption(f"Assigned group: {group_label(assigned)}. Does the page belong here?")
                g_answer = st.pills("Group check", GROUP_ANSWERS, selection_mode="single",
                                    default=prev_g.get("answer") or None,
                                    label_visibility="collapsed", key=f"gcheck_{doc_id}")
                if g_answer == "Wrong":
                    others = [g for g in GROUP_NAMES if g != assigned]
                    prev_other = prev_g.get("reviewer_group") or None
                    st.caption("Which group does it belong in?")
                    g_other = st.pills("Belongs in", others, selection_mode="single",
                                       format_func=group_label,
                                       default=prev_other if prev_other in others else None,
                                       label_visibility="collapsed", key=f"gother_{doc_id}")
            g_note = st.text_input("Group note", value=prev_g.get("note", ""), key=f"gnote_{doc_id}",
                                   placeholder="Optional. Why does it look different?")
            st.divider()

        if st.button("Save and continue", type="primary", key=f"save_{doc_id}"):
            unrated = [s for s, r, w, _, _ in answers.values() if r is None or None in w.values()]
            if unrated:
                st.warning(f"Rate quality, words found and spelling for every reading (missing: Reading "
                           f"{', '.join(map(str, unrated))}).")
            elif n_read > 1 and None in ranks.values():
                st.warning("Rank every reading.")
            elif n_read > 1 and len(set(ranks.values())) != n_read:
                st.warning("Each rank can be used only once (no ties).")
            elif assigned and g_answer is None:
                st.warning("Answer the scan group check (Correct, Wrong or Unsure).")
            elif g_answer == "Wrong" and g_other is None:
                st.warning("Pick the group the page belongs in.")
            else:
                now = datetime.now().isoformat(timespec="seconds")
                save_rows([{"id": doc_id, "position": page.get("position", ""), "engine": e,
                            "reading_slot": s, "rating": r, **w,
                            "issues": "; ".join(i), "note": n.strip(),
                            "rank": ranks.get(e, ""), "n_readings": n_read, "saved_at": now}
                           for e, (s, r, w, i, n) in answers.items()])
                if assigned:
                    save_group_check({"id": doc_id, "position": page.get("position", ""),
                                      "assigned_group": assigned, "answer": g_answer,
                                      "reviewer_group": g_other or "", "note": g_note.strip(),
                                      "saved_at": now})
                if jump_to > 0:
                    st.session_state["_reset_jump"] = True
                st.rerun()

# Summary (engine names revealed only once every page is reviewed)

if current is None and not saved.empty:
    st.divider()
    st.subheader("Results by engine")
    s = saved[saved["engine"].isin(engines)]
    st.dataframe(pd.crosstab(s["engine"], s["rating"]).reindex(columns=list(RATINGS), fill_value=0))
    for col, (label, _) in WORD_QUESTIONS.items():
        if col in s:
            st.caption(label)
            st.dataframe(pd.crosstab(s["engine"], s[col]).reindex(columns=SCALE, fill_value=0))

    if "rank" in s:
        r = s[s["rank"].astype(str).str.isdigit()].copy()
        if not r.empty:
            r["rank"] = r["rank"].astype(int)
            r["n"] = r.groupby("id")["rank"].transform("count")
            r = r[r["n"] > 1]
            # Rank as a 0-1 score (1 = best on that page), so pages with 4 or 5 readings compare.
            r["score"] = (r["rank"] - 1) / (r["n"] - 1)
            st.caption("Ranking, all readings (0-1 score, 1 = ranked best; E is a reference, D-51)")
            st.dataframe(r.groupby("engine")["score"].agg(["mean", "count"]).round(2))
            c = r[~r["engine"].isin(REFERENCE_ENGINES)].copy()
            c["crank"] = c.groupby("id")["rank"].rank(method="first")
            c["cn"] = c.groupby("id")["crank"].transform("count")
            c = c[c["cn"] > 1]
            c["score"] = (c["crank"] - 1) / (c["cn"] - 1)
            st.caption("Ranking, candidates only (E dropped, the rest re-ranked per page)")
            st.dataframe(c.groupby("engine")["score"].agg(["mean", "count"]).round(2))

if current is None and not group_checks.empty:
    st.subheader("Scan group check")
    g = group_checks.assign(assigned_group=group_checks["assigned_group"].map(group_label))
    st.dataframe(pd.crosstab(g["assigned_group"], g["answer"]).reindex(columns=GROUP_ANSWERS, fill_value=0))
