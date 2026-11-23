"""Shared data and UI pieces for the screens: loading the archive, the Reader history, Browse
nodes, search, related records and saved searches."""
from __future__ import annotations

import json
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
import yaml

from pipeline.common import ROOT

state = st.session_state
# Set by screens.use() on every run of app.py
archive = queue = tax = type_info = None


APP = ROOT / "outputs" / "app"


HISTORY = APP / "record_history.csv"


TAXONOMY = ROOT / "taxonomy" / "rim_taxonomy_final.yaml"


EMB = ROOT / "results" / "embeddings"


PER_PAGE = 24


def show_image(path, **kw):
    """Show an image at the column width (older Streamlit versions lack use_container_width)."""
    try:
        st.image(str(path), use_container_width=True, **kw)
    except TypeError:
        st.image(str(path), **kw)


@st.cache_data
def load_static():
    """Load the built archive, the review queue and the taxonomy."""
    archive = pd.read_csv(APP / "archive.csv", dtype=str).fillna("")
    queue = pd.read_csv(APP / "queue.csv", dtype=str).fillna("") if (APP / "queue.csv").exists() else pd.DataFrame()
    tax = yaml.safe_load(TAXONOMY.read_text(encoding="utf-8"))
    return archive, queue, tax


@st.cache_data
def load_vectors():
    """Load the saved page embeddings and their record IDs."""
    ids = pd.read_csv(EMB / "hybrid_qwen3-0.6b_ids.csv", dtype={"id": str})["id"].tolist()
    return ids, np.load(EMB / "hybrid_qwen3-0.6b.npy")


def read_history() -> pd.DataFrame:
    """Return the log of corrections made in the Reader."""
    cols = ["time", "id", "action", "old", "new", "by", "note"]
    if HISTORY.exists():
        return pd.read_csv(HISTORY, dtype=str).fillna("")
    return pd.DataFrame(columns=cols)


def log(record_id, action, old, new, note=""):
    """Append one correction to the Reader history."""
    h = read_history()
    row = {"time": datetime.now().strftime("%Y-%m-%d %H:%M"), "id": record_id, "action": action,
           "old": old, "new": new, "by": st.session_state.user, "note": note}
    pd.concat([h, pd.DataFrame([row])]).to_csv(HISTORY, index=False)


def apply_history(archive: pd.DataFrame) -> pd.DataFrame:
    """Return the archive with the Reader corrections applied."""
    a = archive.copy()
    for h in read_history().itertuples():
        m = a["id"] == h.id
        if h.action == "confirmed":
            a.loc[m, "suggested"] = "False"
        elif h.action == "changed" and h.new in type_info:
            d, dn, s, sn, tn = type_info[h.new]
            a.loc[m, ["department", "department_name", "series", "series_name", "record_type", "record_type_name", "suggested"]] = \
                [d, dn, s, sn, h.new, tn, "False"]
    return a


def go(view, **kw):
    """Switch to an Archive view and set any extra state."""
    state.view = view
    for k, v in kw.items():
        state[k] = v


def node_records(node) -> pd.DataFrame:
    """Return the records in a Browse node: department, series, record type, or a quick-access list."""
    level, key, _ = node
    a = archive
    if level == "department":
        return a[a["department"] == key]
    if level == "series":
        return a[a["series"] == key]
    if level == "type":
        own = a[a["record_type"] == key]
        cross = a[(a["also_fits"] == key) & (a["record_type"] != key)].assign(cross="True")
        return pd.concat([own.assign(cross=""), cross])
    if level == "untyped":
        return a[a["record_type"] == ""]
    if level == "recent":
        return a.sort_values("added", ascending=False).head(24)
    if level == "search":
        return search(key)
    return a


@st.cache_data
def ocr_text(path: str) -> str:
    """Return a page's OCR text, or an empty string if the file is missing."""
    p = ROOT / path
    return p.read_text(encoding="utf-8", errors="replace") if p.exists() else ""


def search(q: str) -> pd.DataFrame:
    """Return records whose title, file ID, Bates number or page text contains q."""
    q = q.strip().lower()
    if not q:
        return archive.iloc[0:0]
    hit = (archive["title"].str.lower().str.contains(q, regex=False)
           | archive["file_id"].str.lower().str.contains(q, regex=False)
           | archive["bates"].str.lower().str.contains(q, regex=False))
    text_hit = archive["text"].map(lambda p: q in ocr_text(p).lower())
    return archive[hit | text_hit]


def open_record(rid, ids):
    """Open a record in the Reader, with ids as the list to step through."""
    go("reader", record=rid, list=list(ids), pdf_page=1)
    st.rerun()


def related(rid: str, k: int = 6) -> pd.DataFrame:
    """Return the k records whose page embeddings are closest to record rid."""
    ids, vecs = load_vectors()
    row = {d: i for i, d in enumerate(ids)}
    if rid not in row:
        return archive.iloc[0:0]
    in_archive = [i for i in archive["id"] if i in row and i != rid]
    sims = vecs[[row[i] for i in in_archive]] @ vecs[row[rid]]
    best = [in_archive[j] for j in np.argsort(-sims)[:k]]
    return archive.set_index("id").loc[best].reset_index()


# Saved searches, kept in outputs/app/saved_searches.json
SAVED = APP / "saved_searches.json"


def read_saved() -> list[dict]:
    """Return the saved searches."""
    try:
        return json.loads(SAVED.read_text(encoding="utf-8"))
    except Exception:
        return []


def write_saved(items: list[dict]):
    """Replace the saved searches."""
    SAVED.write_text(json.dumps(items, indent=2, ensure_ascii=False), encoding="utf-8")


def save_search(item: dict):
    """Save a search for the current user."""
    item.update(saved_by=state.user, saved_at=datetime.now().strftime("%Y-%m-%d %H:%M"),
                sid=datetime.now().strftime("%Y%m%d%H%M%S%f"))
    write_saved(read_saved() + [item])
    st.toast("Saved to Quick access")


def open_saved(item: dict):
    """Reopen a saved question (asked again in its scope) or a saved folder search (filters restored)."""
    state.tab = "Archive"
    if item["kind"] == "question":
        panel = item["panel"]
        if item.get("node"):
            go("browse", node=tuple(item["node"]), shown=PER_PAGE)
        elif item.get("record"):
            go("reader", record=item["record"], list=[item["record"]], pdf_page=1)
        else:
            go("home")
        if item.get("scope"):
            state[f"scope-{panel}"] = item["scope"]
        else:
            state.pop(f"scope-{panel}", None)
        state.pop(f"ans-{panel}", None)
        state[f"start-{panel}"] = item["question"]
    else:
        node = tuple(item["node"]); key = node[1]
        go("browse", node=node, shown=PER_PAGE)
        state[f"in-{key}"] = item.get("within", "")
        state[f"dec-{key}"] = item.get("decade", "All")
        state[f"ord-{key}"] = item.get("order", "Oldest first")
    st.rerun()


def save_popover(where, default_name: str, item: dict, key: str, disabled: bool = False, width="stretch"):
    """Show a "Save this search" button that opens the naming dialog.

    A button is used instead of a popover, which missed the first click right after typing
    in a search box.
    """
    if where.button("Save this search", key=f"savebtn-open-{key}", width=width, disabled=disabled):
        state[f"savename-{key}"] = default_name[:70]
        _save_dialog(item, key)


@st.dialog("Name this search")
def _save_dialog(item: dict, key: str):
    name = st.text_input("Name this search", key=f"savename-{key}", label_visibility="collapsed")
    if st.button("Save", key=f"savebtn-{key}", type="primary", use_container_width=True, disabled=not name.strip()):
        save_search({**item, "name": name.strip()})
        st.rerun()
