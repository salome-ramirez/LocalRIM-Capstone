"""Labeling app: file each sampled page into the taxonomy by hand.

Shows one first page at a time. The labeler picks the department that would keep the
official copy (each with an orienting question), then the series and record type, an
optional second "also fits" type and a certainty; or excludes the page with a reason, or
marks that no type fits. Answers are saved as they are made. Reopening continues at the
first unsaved page; going back shows the saved answer, and the newest save counts.

Modes:
    label_pm    training labels: the sample in manifest order, in batches of 100, until
                250 valid labels -> labels/label_pm_labels_<annotator>.csv
    final_test  blind final-test labels on the pages drawn by draw_final_test.py, until
                150 valid labels -> labels/final_test_labels_<annotator>.csv

After each training save it rebuilds docs/data/site_data.json in the background.

Usage:
    streamlit run pipeline/01_sample/labeler.py
"""

import subprocess
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st
import yaml
from PIL import Image

# Settings

# Paths are relative to the repository root, wherever the app is started from.
REPO = Path(__file__).resolve().parents[2]

LABEL_DIR = REPO / "labels"                   # where queues and labels are saved
PM_DIR = REPO / "data" / "pm_sample"          # Philip Morris sample

# Each mode has its own images and taxonomy file.
#   label_pm   -> the training labels: PM-only sample, taxonomy v3 (final). Output
#                 labels/label_pm_labels.csv. Queue = manifest order (already random).
#   final_test -> the blind final-test labels on the pages drawn by draw_final_test.py.
IMAGE_DIRS = {
    "label_pm": PM_DIR,
    "final_test": PM_DIR,
}
TAXONOMY_FILES = {
    "label_pm": REPO / "taxonomy" / "rim_taxonomy_final.yaml",
    "final_test": REPO / "taxonomy" / "rim_taxonomy_final.yaml",
}
GOOD_LABEL_TARGET = 250   # stop rule: 250 valid labels; excluded pages don't count
BATCH_SIZE = 100          # documents per batch, so helpers can work in parallel
CHECKPOINT = 30           # pause and review after 30 good labels
FINAL_TEST_TARGET = 150   # blind final test: 150 valid labels
FINAL_TEST_PAGES = REPO / "results" / "final_test_pages.csv"

# Exclude reasons: the picked one starts the note, so notes stay consistent.
EXCLUDE_REASONS = [
    "folder cover, no contents shown",
    "blank or unreadable",
    "duplicate of <id>",
    "non-tobacco - <company or brand>",
    "not a business record",
    "Out of scope (HR)",       # too few pages to model: out of scope
]

# The two special choices that aren't record types:
NO_FIT = "NO_FIT"      # a real business record that no type fits (taxonomy gap)
EXCLUDE = "EXCLUDE"    # rule R6: blank, unreadable, or not a business record
NO_SECOND = "(none)"

LABEL_DIR.mkdir(exist_ok=True)

# Letterhead cheat sheet: who usually appears on these pages and which
# department their pages usually go to. Shown only when you open it.
# Each row: (letterhead or addressee, who they were, usually goes to)
LETTERHEADS = [
    ("Council for Tobacco Research (CTR)", "Industry-funded research grant body",
     "R&D; Finance for payments"),
    ("Universities, scientists, laboratories", "External researchers", "R&D"),
    ("Tobacco Institute", "Industry trade and public affairs association",
     "Corporate affairs"),
    ("Congress, FDA, state agencies, press", "Government and media",
     "Corporate affairs"),
    ("Law firms", "Outside counsel", "Legal; Finance for invoices"),
    ("Advertising agencies, brands", "Advertising and brand work", "Marketing"),
    ("Retailers, wholesalers, distributors", "Trade customers",
     "Marketing (Retailer and wholesaler dealings)"),
    ("Suppliers and vendors", "Materials, equipment and services",
     "Finance; Quality for specifications and safety data sheets"),
    ("Consumers", "Consumer correspondence",
     "Marketing; Quality for product complaints"),
    ("Unions, job applicants", "Employees and applicants", "Human resources"),
    ("Competitors (RJR, Lorillard, B&W)", "Competitive analysis",
     "Marketing (Competitor research); R&D for product analyses"),
    ("Market research firms", "Consumer surveys and focus groups",
     "Marketing (Consumer surveys and research)"),
    ("Kraft, General Foods, Miller", "Non-tobacco businesses",
     "Exclude (R8)"),
    ("Payables documents", "Invoices, checks, purchase orders, quotes",
     "Finance (R9)"),
]


# Taxonomy: read the nested YAML and flatten it into one row per record type

def as_text(value) -> str:
    """Join a list field (includes, excludes) into one line."""
    if isinstance(value, list):
        return "; ".join(str(v) for v in value)
    return value or ""


# st.cache_data stores the result so the YAML isn't re-read on every click.
# Passing the file's modified time (mtime) as an argument means the cache
# refreshes automatically whenever you save changes to the YAML.
@st.cache_data
def load_taxonomy(path_str: str, mtime: float):
    """Read the taxonomy and flatten it to one row per record type."""
    data = yaml.safe_load(Path(path_str).read_text(encoding="utf-8"))
    rows = []
    # question and hints are new in v2; .get() keeps older YAMLs working.
    functions = [{"id": f["id"], "name": f["name"],
                  "definition": f.get("definition", ""),
                  "question": f.get("question", ""),
                  "hints": f.get("hints", [])}
                 for f in data["functions"]]
    # Everything is shown A to Z: departments by name, then types by name.
    functions = sorted(functions, key=lambda f: f["name"].lower())
    for function in data["functions"]:
        for series in function.get("series", []):
            for rtype in series.get("types", []):
                rows.append({
                    "type_id": rtype["id"],
                    "type_name": rtype["name"],
                    "series_id": series["id"],
                    "series_name": series["name"],
                    "function_id": function["id"],
                    "function_name": function["name"],
                    "definition": rtype.get("definition", ""),
                    "looks_like": rtype.get("looks_like", ""),
                    "includes": as_text(rtype.get("includes")),
                    "excludes": as_text(rtype.get("excludes")),
                })
    table = pd.DataFrame(rows)
    # Department, then series, then type, A to Z: tabs stay grouped by series.
    table = table.sort_values(
        ["function_name", "series_name", "type_name"],
        key=lambda col: col.str.lower()).reset_index(drop=True)
    rules = data.get("labeling_rules", [])
    # Type IDs must be unique, because the type alone identifies its parents.
    duplicates = table.loc[table["type_id"].duplicated(), "type_id"].tolist()
    return table, functions, rules, duplicates, data.get("version")


# Queue: the list of documents for each mode

def get_queue(mode: str) -> list[str]:
    """Return the document queue for a mode."""
    if mode == "final_test":
        # The drawn test pages, in the shuffled order saved by draw_final_test.py.
        pages = pd.read_csv(FINAL_TEST_PAGES, dtype=str)
        return pages["image"].tolist()
    if mode == "label_pm":
        # The manifest is already in random draw order. Reading it each time means documents
        # added later join the end.
        # Pages flagged by the duplicate check (duplicates.csv) are skipped.
        manifest = pd.read_csv(PM_DIR / "manifest.csv", dtype=str)
        dup_file = PM_DIR / "duplicates.csv"
        if dup_file.exists():
            dups = set(pd.read_csv(dup_file, dtype=str)["id"])
            manifest = manifest[~manifest["id"].isin(dups)]
        return manifest["image"].tolist()


def read_saved(out_file: Path) -> pd.DataFrame:
    """Read the saved labels, keeping each document's newest answer."""
    if not out_file.exists():
        return pd.DataFrame()
    saved = pd.read_csv(out_file, dtype=str, keep_default_na=False)
    # If a document was saved twice (you went back and changed it),
    # keep only the most recent answer.
    return saved.drop_duplicates(subset="path", keep="last")


def save_row(out_file: Path, row: dict):
    """Append one answer to its label file."""
    pd.DataFrame([row]).to_csv(
        out_file, mode="a", header=not out_file.exists(), index=False
    )
    if out_file.name.startswith(("label_pm_labels_", "skipped_pm_")):
        refresh_site()   # the site shows training labels only; test labels stay off it


def refresh_site():
    """Rebuild the site data in the background after a save, so labeling never waits."""
    script = REPO / "pipeline" / "06_site_data" / "build_site_data.py"
    if script.exists():
        try:
            subprocess.Popen([sys.executable, str(script)], cwd=script.parent,
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except OSError:
            pass   # the site can always be rebuilt by hand at wrap-up


# Page layout

st.set_page_config(page_title="LocalRIM · Records Classification", layout="wide")

# Image view. "Fit page to screen" on: the image column stays pinned
# while you scroll the choices, and the whole page fits the screen height.
# OFF: the image shows at full column width to read small print, and scrolls
# with the page. Streamlit's own full-screen button is hidden because it
# clashed with the pinned column (the choices showed through, no way to close).
# CSS position: sticky  https://developer.mozilla.org/en-US/docs/Web/CSS/position#sticky
fit_to_screen = st.sidebar.checkbox(
    "Fit page to window", value=True,
    help="Clear to view the page at full width.")

style = """
[data-testid="stElementToolbar"], [data-testid="StyledFullScreenButton"] {
    display: none !important;
}
/* Primary choice (department, series, record type): red outline in the same
   red as "Save and continue", so the eye lands there first (Nov 7). */
.st-key-primary_box {
    background: #FFFFFF;
    border: 2px solid #FF4B4B;
    border-radius: 10px;
    padding: 10px 14px;
}
/* Secondary classification: same single box, light gray (Nov 7; was bright yellow) */
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
/* A selected secondary tile turns darker gray, so a pick stays visible */
.st-key-alsofits_box button[data-variant="pills"][aria-checked="true"],
.st-key-alsofits_box button[data-variant="pills"][data-selected="true"] {
    background: #E4E4E7 !important;
    border-color: #8A8A93 !important;
    color: #2E2E33 !important;
}
/* Tiles wrap onto new lines instead of scrolling sideways */
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

mode = st.sidebar.radio("Dataset", ["label_pm", "final_test"],
                        format_func=lambda m: {"label_pm": "Labeling set (PM sample)",
                                               "final_test": "Final test (blind)"}[m])
taxonomy_file = TAXONOMY_FILES[mode]
IMAGE_DIR = IMAGE_DIRS[mode]

if not taxonomy_file.exists():
    st.error(f"File not found: {taxonomy_file.relative_to(REPO)}")
    st.stop()
if not IMAGE_DIR.exists():
    st.error(f"Image folder not found: {IMAGE_DIR}")
    st.stop()

taxonomy, functions, rules, duplicates, version = load_taxonomy(
    str(taxonomy_file), taxonomy_file.stat().st_mtime
)
if duplicates:
    st.error(f"Duplicate record type IDs in the taxonomy file: {duplicates}")
    st.stop()

types_by_id = taxonomy.set_index("type_id").to_dict("index")
type_ids = taxonomy["type_id"].tolist()

if mode == "final_test" and not FINAL_TEST_PAGES.exists():
    st.error("Final-test pages not drawn yet. Run: python draw_final_test.py")
    st.stop()
full_queue = get_queue(mode)
offset = 0
if mode == "label_pm":
    # Batches of 100 so several people can label in parallel. Each labeler
    # saves to their own file, so nobody overwrites anyone; files merge later.
    labeler_name = st.sidebar.text_input("Annotator", value="salo").strip().lower() or "salo"
    n_batches = max(1, -(-len(full_queue) // BATCH_SIZE))
    batch = st.sidebar.selectbox(
        "Batch", list(range(1, n_batches + 1)),
        format_func=lambda b: (f"Batch {b} (documents {(b - 1) * BATCH_SIZE + 1}"
                               f"–{min(b * BATCH_SIZE, len(full_queue))})"))
    offset = (batch - 1) * BATCH_SIZE
    queue = full_queue[offset:offset + BATCH_SIZE]
    out_file = LABEL_DIR / f"label_pm_labels_{labeler_name}.csv"
elif mode == "final_test":
    # One list, no batches; its own label file, never mixed with the training labels.
    labeler_name = st.sidebar.text_input("Annotator", value="salo").strip().lower() or "salo"
    queue = full_queue
    out_file = LABEL_DIR / f"final_test_labels_{labeler_name}.csv"
saved = read_saved(out_file)
done = set(saved["path"]) if not saved.empty else set()

# Skip for now: documents set aside to come back to at the end of the
# session (e.g. a failed download). Kept in their own file, so they never count
# as labeled or excluded. Labeling a skipped document takes it off the list.
skip_file = (LABEL_DIR / f"skipped_pm_{labeler_name}.csv" if mode == "label_pm"
             else LABEL_DIR / f"skipped_final_test_{labeler_name}.csv")
skips = read_saved(skip_file) if skip_file else pd.DataFrame()
skip_reason = {} if skips.empty else dict(zip(skips["path"], skips["reason"]))
skipped = [p for p in full_queue if p in skip_reason and p not in done]

# --- Sidebar: progress, navigation and the labeling rules ---
st.sidebar.caption(f"Taxonomy version {version} · {taxonomy_file.name}")
done_count = sum(p in done for p in queue)
st.sidebar.write(f"**{done_count} / {len(queue)}** reviewed in this batch")
if mode == "label_pm":
    # Good labels = labeled (not excluded, not no-fit), counted across every
    # labeler's file. The stop rule counts these.
    all_files = [read_saved(f) for f in LABEL_DIR.glob("label_pm_labels_*.csv")]
    all_saved = pd.concat([s for s in all_files if not s.empty], ignore_index=True) \
        if any(not s.empty for s in all_files) else pd.DataFrame()
    good = 0 if all_saved.empty else int(
        (all_saved.drop_duplicates(subset="path", keep="last")["outcome"] == "labeled").sum())
    st.sidebar.write(f"**{good} / {GOOD_LABEL_TARGET}** valid labels")
    st.sidebar.progress(min(good / GOOD_LABEL_TARGET, 1.0))
    if good >= GOOD_LABEL_TARGET:
        st.sidebar.success(f"Target reached: {GOOD_LABEL_TARGET} valid labels.")
    elif good >= CHECKPOINT and not (LABEL_DIR / "checkpoint_done.txt").exists():
        st.sidebar.warning(f"Checkpoint: {CHECKPOINT} valid labels. Review results before continuing.")
elif mode == "final_test":
    test_files = [read_saved(f) for f in LABEL_DIR.glob("final_test_labels_*.csv")]
    test_saved = pd.concat([s for s in test_files if not s.empty], ignore_index=True) \
        if any(not s.empty for s in test_files) else pd.DataFrame()
    good = 0 if test_saved.empty else int(
        (test_saved.drop_duplicates(subset="path", keep="last")["outcome"] == "labeled").sum())
    st.sidebar.write(f"**{good} / {FINAL_TEST_TARGET}** valid test labels")
    st.sidebar.progress(min(good / FINAL_TEST_TARGET, 1.0))
    if good >= FINAL_TEST_TARGET:
        st.sidebar.success(f"Target reached: {FINAL_TEST_TARGET} valid test labels. Stop here.")

review_skipped = False
if skip_file:
    st.sidebar.write(f"**Skipped: {len(skipped)}**")
    review_skipped = st.sidebar.checkbox("Review skipped only", value=False,
                                         disabled=not skipped)

st.sidebar.caption(
    "Set to 0 to resume at the next unlabeled document, "
    "or enter a position to review a saved label."
)
# Back to where you left off: after re-saving an earlier document, or
# with the button, the position resets to 0 (the next unlabeled document).
if st.session_state.pop("_reset_jump", False) or "jump_to" not in st.session_state:
    st.session_state["jump_to"] = 0
jump_to = st.sidebar.number_input("Go to position in batch", min_value=0,
                                  max_value=len(queue), step=1, key="jump_to")
st.sidebar.button("Back to where I left off", disabled=jump_to == 0,
                  on_click=lambda: st.session_state.update(jump_to=0))

# Closed by default to keep the screen calm; the rules that matter for
# your chosen department also appear under "Can't decide?".
with st.sidebar.expander("Labeling rules", expanded=False):
    for r in rules:
        st.markdown(f"**{r['id']}** {r['rule']}")

if review_skipped:
    # Go through only the skipped documents, in labeling order, across batches.
    queue = skipped
    offset = 0
    current_index = 0 if skipped else None
    if not skipped:
        st.success("No skipped documents left.")
elif jump_to > 0:
    current_index = jump_to - 1
else:
    remaining = [i for i, p in enumerate(queue) if p not in done and p not in skip_reason]
    if not remaining:
        st.success("All documents in this batch are labeled or skipped.")
        current_index = None
    else:
        current_index = remaining[0]


# --- Main area: image on the left, choices on the right ---
if current_index is not None:
    current_path = queue[current_index]
    left, right = st.columns([3, 2])

    with left:
        # The file name is saved with the label (path, doc_id) but not shown,
        # so it can't nudge the label.
        header = f"Document {full_queue.index(current_path) + 1} of {len(full_queue)}"
        if mode == "label_pm":
            header += f" · {good} of {GOOD_LABEL_TARGET} valid labels"
        elif mode == "final_test":
            header = (f"Test page {queue.index(current_path) + 1} of {len(queue)}"
                      f" · {good} of {FINAL_TEST_TARGET} valid test labels")
        st.subheader(header)
        image = Image.open(IMAGE_DIR / current_path).convert("RGB")
        st.image(image, width="stretch")

    with right:
        # Two-step choice: first the department (8 options, each with an
        # orienting question), then only the record types inside it (1 or 2).
        # Every widget's key includes the document path, so each new
        # document starts blank.
        k = current_path
        function_by_id = {f["id"]: f for f in functions}
        step1_options = [f["id"] for f in functions] + [NO_FIT, EXCLUDE]

        # If this document was already saved, fill the form with the most
        # recent answer, so going back shows what you chose.
        # Saving again adds a new row; the newest one is the one that counts.
        prev = {}
        if not saved.empty and current_path in done:
            prev = saved.loc[saved["path"] == current_path].iloc[-1].to_dict()
            st.info(f"Already saved ({prev.get('saved_at', '')}). Your answer is "
                    "shown below. Change it and press Save to replace it, or set "
                    "'Go to document #' to 0 to move on.")
        if prev.get("outcome") == "labeled":
            prev_step1 = prev.get("function_id")
        elif prev.get("outcome") == "no_fit":
            prev_step1 = NO_FIT
        elif prev.get("outcome") == "exclude":
            prev_step1 = EXCLUDE
        else:
            prev_step1 = None

        def describe_function(choice: str) -> str:
            if choice == NO_FIT:
                return "No fit"
            if choice == EXCLUDE:
                return "Exclude"
            return function_by_id[choice]["name"]

        def function_caption(choice: str) -> str:
            if choice == NO_FIT:
                return "Business record outside the taxonomy"
            if choice == EXCLUDE:
                return "Blank, unreadable or not a business record"
            return function_by_id[choice]["question"]

        # The orienting sentence sits above the list; each department shows
        # its own question underneath its name.
        st.markdown("**Office of record:** which Philip Morris department would hold the official copy?")
        st.caption("Classify by the document's purpose, not its format.")
        # Closed until you click it. The letterhead is a starting point;
        # what the page does still decides the department.
        with st.expander("Reference: senders and typical departments"):
            st.table(pd.DataFrame(
                sorted(LETTERHEADS, key=lambda row: row[0].lower()),
                columns=["Sender or addressee", "Description",
                         "Typical department"],
            ).set_index("Sender or addressee"))
            st.caption("The sender is a starting point; the document's purpose "
                       "determines the department.")
        with st.expander("Reference: department definitions"):
            for c in step1_options:
                st.markdown(f"**{describe_function(c)}**: {function_caption(c)}")
        # Filing tiles: Drawer (department) -> Folder (series)
        # -> Tab. Each row only appears after the one above is picked.
        # st.pills docs: https://docs.streamlit.io/develop/api-reference/widgets/st.pills
        # Short names on the tiles so all 10 fit; the breadcrumb shows full names.
        short = {"RND": "R&D", "QSH": "Quality", "MKT": "Marketing",
                 "AFF": "Corp. affairs", "LEG": "Legal", "FIN": "Finance",
                 "HR": "HR", "ADM": "Admin", NO_FIT: "No fit", EXCLUDE: "Exclude"}
        with st.container(key="primary_box"):
            st.caption("Department")
            step1 = st.pills("Department", step1_options, selection_mode="single",
                             default=prev_step1 if prev_step1 in step1_options else None,
                             format_func=lambda c: short.get(c, describe_function(c)),
                             label_visibility="collapsed", key=f"fn_{k}")
            if step1:
                st.caption(function_caption(step1))

            primary = None
            exclude_reason = None
            if step1 in function_by_id:
                fn = function_by_id[step1]
                options = taxonomy[taxonomy["function_id"] == step1]

                # 2. Series (folder) inside the department
                series = (options[["series_id", "series_name"]].drop_duplicates()
                          .set_index("series_id")["series_name"].to_dict())
                series_ids = list(series)
                prev_series = prev.get("series_id")
                st.caption("Series")
                step2 = st.pills(
                    "Series", series_ids, selection_mode="single",
                    default=(prev_series if prev_series in series_ids
                             else (series_ids[0] if len(series_ids) == 1 else None)),
                    format_func=lambda s: series[s],
                    label_visibility="collapsed", key=f"series_{k}")

                # 3. Tab (record type) inside the folder
                if step2:
                    tabs = options[options["series_id"] == step2]
                    tab_ids = tabs["type_id"].tolist()
                    prev_type = prev.get("type_id")
                    st.caption("Record type")
                    primary = st.pills(
                        "Tab", tab_ids, selection_mode="single",
                        default=(prev_type if prev_type in tab_ids
                                 else (tab_ids[0] if len(tab_ids) == 1 else None)),
                        format_func=lambda t: types_by_id[t]["type_name"],
                        label_visibility="collapsed", key=f"type_{k}")
                    if primary:
                        # Breadcrumb + what this tab usually looks like
                        st.markdown(f"**{fn['name']} › {series[step2]} › "
                                    f"{types_by_id[primary]['type_name']}**")
                        st.caption(f"Typical documents: {types_by_id[primary]['looks_like']}")
                    with st.expander("Record types in this series"):
                        for r in tabs.itertuples():
                            st.markdown(f"**{r.type_name}**: {r.looks_like or r.definition}")
                # Tie-breakers for this department only, closed until needed.
                with st.expander("Classification guidance"):
                    for hint in fn["hints"]:
                        st.markdown(f"- {hint}")
                    st.caption("If uncertain, select the closest record type, mark it as "
                               "uncertain, and record the alternative as a secondary classification.")
            elif step1 == NO_FIT:
                primary = NO_FIT
                st.caption("Describe the document in the note. No-fit records identify gaps in the taxonomy.")
            elif step1 == EXCLUDE:
                primary = EXCLUDE
                # The reason is saved at the start of the note (R7, R8, R10).
                exclude_reason = st.selectbox(
                    "Exclude reason", EXCLUDE_REASONS, index=None,
                    placeholder="Select a reason", key=f"exreason_{k}")
                st.caption("For duplicates or non-tobacco records, add the document ID or company in the note.")

        st.divider()
        # Also fits: same Drawer -> Folder -> Tab tiles as above, in a yellow
        # box with yellow selected tiles so it reads as the second choice.
        prev_second = prev.get("also_fits") or ""
        prev_t = types_by_id.get(prev_second, {})
        second = NO_SECOND
        with st.container(key="alsofits_box"):
            st.markdown("**Secondary classification** (optional)")
            a_fn_options = [f["id"] for f in functions]
            a_fn = st.pills("Also fits drawer", a_fn_options, selection_mode="single",
                            default=prev_t.get("function_id"),
                            format_func=lambda c: short.get(c, c),
                            label_visibility="collapsed", key=f"afn_{k}")
            if a_fn:
                a_opts = taxonomy[taxonomy["function_id"] == a_fn]
                a_series = (a_opts[["series_id", "series_name"]].drop_duplicates()
                            .set_index("series_id")["series_name"].to_dict())
                a_series_ids = list(a_series)
                a_s = st.pills("Also fits folder", a_series_ids, selection_mode="single",
                               default=(prev_t.get("series_id") if prev_t.get("series_id") in a_series_ids
                                        else (a_series_ids[0] if len(a_series_ids) == 1 else None)),
                               format_func=lambda s: a_series[s],
                               label_visibility="collapsed", key=f"aseries_{k}")
                if a_s:
                    a_tab_ids = a_opts[a_opts["series_id"] == a_s]["type_id"].tolist()
                    a_t = st.pills("Also fits tab", a_tab_ids, selection_mode="single",
                                   default=(prev_second if prev_second in a_tab_ids
                                            else (a_tab_ids[0] if len(a_tab_ids) == 1 else None)),
                                   format_func=lambda t: types_by_id[t]["type_name"],
                                   label_visibility="collapsed", key=f"atype_{k}")
                    if a_t:
                        second = a_t
                        st.caption(f"{types_by_id[a_t]['function_name']} › "
                                   f"{a_series[a_s]} › {types_by_id[a_t]['type_name']}")
            st.caption("Select the active department again to clear.")
        unsure = st.checkbox("Mark as uncertain",
                             value=prev.get("certainty") == "unsure",
                             key=f"unsure_{k}")
        note = st.text_input(
            "Note",
            value=prev.get("note", ""),
            placeholder="Required for uncertain, no-fit and excluded records. "
                        "Describe the deciding evidence.",
            key=f"note_{k}",
        )
        if current_path in skip_reason and current_path not in done:
            st.info(f"Skipped: {skip_reason[current_path]}")
        b1, b2 = st.columns([1, 1])
        submitted = b1.button("Save and continue", type="primary", key=f"save_{k}")
        if skip_file and b2.button("Skip for now", type="secondary", key=f"skip_{k}"):
            st.session_state[f"skipping_{k}"] = True
        if skip_file and st.session_state.get(f"skipping_{k}"):
            with st.container(border=True):
                why = st.radio("Why skip this one?",
                               ["Download failed / page not shown",
                                "Need to look at it later", "Other"],
                               key=f"skipwhy_{k}")
                other = st.text_input("Other reason", key=f"skipother_{k}") if why == "Other" else ""
                c1, c2 = st.columns([1, 1])
                if c1.button("Confirm skip", key=f"skipok_{k}"):
                    reason = other.strip() if why == "Other" else why
                    if not reason:
                        st.warning("Add a short reason for the skip.")
                    else:
                        save_row(skip_file, {
                            "path": current_path,
                            "doc_id": Path(current_path).stem,
                            "reason": reason,
                            "saved_at": datetime.now().isoformat(timespec="seconds"),
                        })
                        st.session_state[f"skipping_{k}"] = False
                        st.rerun()
                if c2.button("Cancel", key=f"skipcancel_{k}"):
                    st.session_state[f"skipping_{k}"] = False
                    st.rerun()

        if submitted:
            # Exclude: the picked reason starts the note.
            if primary == EXCLUDE and exclude_reason:
                reason = exclude_reason.split(" <")[0].rstrip(" -")
                if not note.strip().startswith(reason):
                    note = f"{reason} | {note}" if note.strip() else reason
            needs_note = unsure or primary in (NO_FIT, EXCLUDE)
            if primary is None:
                st.warning("Select a department and record type, or choose No fit or Exclude.")
            elif needs_note and not note.strip():
                st.warning("A note is required for uncertain, no-fit and excluded records.")
            else:
                is_type = primary not in (NO_FIT, EXCLUDE)
                t = types_by_id[primary] if is_type else {}
                save_row(out_file, {
                    "path": current_path,
                    "doc_id": Path(current_path).stem,
                    "type_id": primary if is_type else "",
                    "type_name": t.get("type_name", ""),
                    "series_id": t.get("series_id", ""),
                    "function_id": t.get("function_id", ""),
                    "outcome": "labeled" if is_type else primary.lower(),
                    "also_fits": "" if second == NO_SECOND else second,
                    "certainty": "unsure" if unsure else "sure",
                    "note": note.strip(),
                    "taxonomy_file": taxonomy_file.name,
                    "saved_at": datetime.now().isoformat(timespec="seconds"),
                })
                # A revisited document: go back to the next unlabeled one.
                if jump_to > 0:
                    st.session_state["_reset_jump"] = True
                st.rerun()

        with st.expander("Reference: full taxonomy"):
            for _, r in taxonomy.iterrows():
                st.markdown(f"**{r['function_name']} › {r['type_name']}**  \n"
                            f"{r['definition']}")
                if r["looks_like"]:
                    st.caption(f"Typical documents: {r['looks_like']}")
                if r["includes"]:
                    st.caption(f"Includes: {r['includes']}")
                if r["excludes"]:
                    st.caption(f"Excludes: {r['excludes']}")

# Summary of what you've saved — use this for the taxonomy revision

if not saved.empty:
    st.divider()
    with st.expander("Label summary",
                     expanded=current_index is None):
        labeled = saved[saved["outcome"] == "labeled"]
        n = len(saved)
        st.write(f"{n} saved: {len(labeled)} labeled, "
                 f"{(saved['outcome'] == 'no_fit').sum()} no fit, "
                 f"{(saved['outcome'] == 'exclude').sum()} excluded, "
                 f"{(saved['certainty'] == 'unsure').sum()} unsure.")

        st.markdown("**How often each type was used.**")
        counts = (labeled["type_id"].value_counts()
                  .reindex(type_ids, fill_value=0).rename("count").to_frame())
        counts["type_name"] = [types_by_id[t]["type_name"] for t in counts.index]
        st.dataframe(counts.sort_values("count"))

        st.markdown("**Primary and secondary classification pairs**")
        pairs = saved[saved["also_fits"] != ""]
        if pairs.empty:
            st.write("None recorded.")
        else:
            st.dataframe(pairs.groupby(["type_id", "also_fits"]).size()
                         .rename("count").sort_values(ascending=False))

        st.markdown("**No-fit records**")
        st.dataframe(saved.loc[saved["outcome"] == "no_fit", ["path", "note"]])

        st.markdown("**Excluded records**")
        st.dataframe(saved.loc[saved["outcome"] == "exclude", ["path", "note"]])

        st.markdown("**Uncertain labels**")
        st.dataframe(saved.loc[saved["certainty"] == "unsure",
                               ["path", "type_id", "also_fits", "note"]])
