"""Add files and the Intake screen: batch inventory and approval."""
from __future__ import annotations

import html
from datetime import datetime

import pandas as pd
import streamlit as st

from pipeline.common import ROOT
from localrim import intake as intake_db
from .shared import show_image

state = st.session_state
# Set by screens.use() on every run of app.py
archive = queue = dept_name = type_info = TYPE_LABEL = LABEL_TYPE = None


@st.cache_resource(show_spinner=False)
def pipeline():
    """Load the intake pipeline once per server process."""
    from localrim import pipeline as localrim_pipeline
    return localrim_pipeline.Pipeline(rag=True)


@st.dialog("Add files", width="large")
def add_files():
    """Upload dialog: read and file each document from its first page, then link to the batch in Intake."""
    if state.get("just_added"):
        b = state.just_added
        rows = intake_db.read(intake_db.INTAKE).query("batch == @b")
        ready = (rows["route"] == "Ready to approve").sum()
        with st.container(key="zone-added"):
            st.success(f"Batch {b}: {len(rows)} documents read · {ready} ready to approve · {len(rows) - ready} need review")
            if st.button("Go to this batch in Intake", type="primary", use_container_width=True):
                state.tab, state.intake_batch, state.just_added = "Intake", b, None
                st.rerun()
        return
    files = st.file_uploader("Drag scanned documents here, or browse your computer", type=["png", "pdf"],
                             accept_multiple_files=True, help="Each document is filed from its first page.")
    t = datetime.now()
    st.caption(f"Uploaded by {state.user} · {t:%B} {t.day}, {t:%Y %H:%M}")
    if files and st.button(f"Read and file {len(files)} document{'s' if len(files) > 1 else ''}", type="primary",
                           use_container_width=True):
        with st.spinner("Starting the reading tools (first time only, about 40 seconds)…"):
            pipe = pipeline()
        batch = intake_db.next_batch()
        known = intake_db.known_file_ids(archive, queue)
        rows, vecs = [], []
        bar = st.progress(0.0)
        with st.status(f"Batch {batch}: reading and filing…", expanded=True) as status:
            for k, f in enumerate(files, 1):
                st.write(f"{f.name}: reading the page, filing it, suggesting a title and date…")
                row, vec = intake_db.process_page(pipe, batch, k, f.name, f.getvalue(), known, type_info, dept_name)
                rows.append(row)
                vecs.append(vec)
                st.write(f"→ {row.get('route', 'Needs review')}"
                         + (f" · {row['route_reason']}" if row.get("route_reason") else "")
                         + (f" · {row.get('department_name', '')} {float(row['confidence']):.0%}" if row.get("confidence") else ""))
                bar.progress(k / len(files))
            status.update(label=f"Batch {batch}: done", state="complete")
        intake_db.save_batch(batch, rows, vecs, state.user)
        from localrim.assistant import reset_vectors
        reset_vectors()
        state.just_added = str(batch)
        st.rerun(scope="fragment")


# One bar colour for both themes. The table is drawn on a canvas CSS can't reach, and Streamlit
# re-themes it before the page, so a theme-dependent colour could disappear. Contrast is about
# 4:1 on both the light and the dark table.
BAR = "#8F7B70"


def thumb_uri(path: str) -> str:
    """Return a thumbnail as a data URI for table image columns, or an empty string."""
    import base64
    p = ROOT / path
    return "data:image/jpeg;base64," + base64.b64encode(p.read_bytes()).decode() if path and p.exists() else ""


def when(stamp: str) -> str:
    """Format a 'YYYY-MM-DD HH:MM' stamp for display."""
    try:
        t = datetime.strptime(stamp, "%Y-%m-%d %H:%M")
        return f"{t:%B} {t.day}, {t:%Y} at {t:%H:%M}"
    except ValueError:
        return stamp


# Quick view: one document at a time with its full page, title, file ID and the file name it
# will be saved as. state.qv = {"src": "ready" | "review", "i": n}.
def _qv_dismiss():
    state.pop("qv", None)
    state.nr_nonce = state.get("nr_nonce", 0) + 1          # new table key, so the same Needs review row can be opened again


def quick_view(docs: list[dict]):
    """Show the documents one at a time in a dialog."""
    i = max(0, min(state.qv.get("i", 0), len(docs) - 1))
    d = docs[i]

    @st.dialog(f"Document {i + 1} of {len(docs)}", width="large", on_dismiss=_qv_dismiss)
    def _window():
        bar = st.container(key="qv-bar", horizontal=True, vertical_alignment="center", gap="small")
        if bar.button("‹", disabled=i == 0, key="qv-prev"):
            state.qv["i"] = i - 1; st.rerun()
        bar.markdown(f'<div class="rim-card-h">{html.escape(d["title"])}</div><div class="rim-muted rim-mono">{html.escape(d["file_id"])}</div>',
                     unsafe_allow_html=True, width="stretch")
        if bar.button("›", disabled=i >= len(docs) - 1, key="qv-next"):
            state.qv["i"] = i + 1; st.rerun()
        img = ROOT / d["image"] if d["image"] else None
        if img is not None and img.exists():
            show_image(img)
        elif d.get("thumb") and (ROOT / d["thumb"]).exists():
            show_image(ROOT / d["thumb"])
        st.markdown(f'<div class="rim-eyebrow" style="margin-top:8px">Will be saved as</div><div class="rim-file">{html.escape(d["file_name"])}</div>',
                    unsafe_allow_html=True)
    _window()


def intake_screen():
    """Intake screen: the selected batch, its pages ready to approve and those needing review."""
    batches = intake_db.read(intake_db.BATCHES)
    if batches.empty:
        st.markdown('<div class="rim-h1" role="heading" aria-level="1">Intake</div>', unsafe_allow_html=True)
        st.markdown('<div class="rim-empty">No batches yet. Add scanned documents with <b>⬆ Add files</b>; each batch waits here until a person approves it.</div>',
                    unsafe_allow_html=True)
        return
    rows_all = intake_db.read(intake_db.INTAKE)
    order = batches["batch"].astype(int).sort_values(ascending=False).astype(str).tolist()
    if state.get("intake_batch") not in order:
        state.intake_batch = order[0]

    def label(b):
        x = batches[batches["batch"] == b].iloc[0]
        w = ((rows_all["batch"] == b) & (rows_all["status"] == "waiting")).sum()
        return f"Batch {b} · {when(x['uploaded_at'])} · {x['pages']} pages" + (f" · {w} waiting" if w else "")

    _inbar = st.container(key="zone-inbar", horizontal=True, vertical_alignment="bottom", gap="medium")
    head = _inbar.container(key="inbar-head")              # the batch picker drops under the title when there's no room
    b = _inbar.selectbox("Batch", order, index=order.index(state.intake_batch), format_func=label, width=460)
    state.intake_batch = b
    info = batches[batches["batch"] == b].iloc[0]
    rows = rows_all[rows_all["batch"] == b]
    waiting = rows[rows["status"] == "waiting"]
    flagged = rows[rows["route"] == "Needs review"]
    approved = rows[rows["status"] == "approved"]
    head.markdown(f'<div class="rim-eyebrow">Intake</div><div class="rim-h1" role="heading" aria-level="1">Batch {b}</div>'
                    f'<div class="rim-muted">Uploaded by <b>{html.escape(info["uploaded_by"])}</b> on {when(info["uploaded_at"])} · '
                    f'{len(rows)} documents · {(rows["route"] == "Ready to approve").sum()} ready to approve · '
                    f'{len(flagged)} need review</div>', unsafe_allow_html=True)
    if info["approved_at"]:
        c1, c2 = st.columns([4, 1.2], vertical_alignment="center")
        c1.success(f"Approved by {info['approved_by']} on {when(info['approved_at'])} · {info['approved']} filed to the Archive")
        if intake_db.can_undo(b) and c2.button("Undo approval", use_container_width=True,
                                                help=f"Available for {intake_db.UNDO_MINUTES} minutes after approving"):
            intake_db.undo(b, state.user)
            st.rerun()

    with st.container(key="zone-inready"):
        st.markdown('<div class="rim-h2" role="heading" aria-level="2" style="margin-top:14px">Ready to approve</div>', unsafe_allow_html=True)
        if waiting.empty:
            st.caption("Nothing waiting in this batch." if len(approved) else "No document in this batch was confident enough to file without a person.")
        else:
            st.caption("Filed by the pipeline with confidence at or above 55%, the cutoff set so filed pages are right about 90% "
                       "of the time. Check the title and date, change "
                       "anything that's wrong, then approve. Approving confirms the department; record types stay "
                       "\"suggested\" unless you change them.")
            if st.button("View pages", key=f"qv-open-{b}"):
                state.qv = {"src": "ready", "i": 0}
            w = waiting.reset_index(drop=True)
            table = pd.DataFrame({
                "approve": True,
                "page": w["thumb"].map(thumb_uri),
                "title": w["title"],
                "file_id": w["file_id"],
                "date": w["date_iso"],
                "confidence": pd.to_numeric(w["confidence"], errors="coerce"),
                "record_type": w["record_type"].map(lambda t: TYPE_LABEL.get(t, "")),
            })
            table.index = w["id"].tolist()
            edited = st.data_editor(
                table, key=f"ed-{b}", use_container_width=True, hide_index=True, row_height=90,
                column_config={
                    # Fixed widths; text wraps inside the 90 px rows. There is no Department column: the record
                    # type starts with it and changing the type changes the department on approval. Approve, Page
                    # and Title are pinned, so at narrow widths the rest scrolls inside the table.
                    "approve": st.column_config.CheckboxColumn("Approve", width=82, pinned=True),
                    "page": st.column_config.ImageColumn("Page", width=64, pinned=True),
                    "title": st.column_config.TextColumn("Title (suggested)", width=136, pinned=True),
                    "file_id": st.column_config.TextColumn("File ID", disabled=True, width=128),
                    "date": st.column_config.TextColumn("Date (suggested)", width=140,
                                                        help="Document date: YYYY-MM-DD, YYYY-MM, YYYY or undated"),
                    "confidence": st.column_config.ProgressColumn("Confidence", format="percent", min_value=0, max_value=1,
                                                                  width=84, color=BAR),
                    "record_type": st.column_config.SelectboxColumn("Record type (suggested)", options=[""] + list(LABEL_TYPE), width=496),
                })
            if state.get("qv", {}).get("src") == "ready":
                quick_view([{"title": e.title, "file_id": e.file_id, "image": w.loc[k, "image"], "thumb": w.loc[k, "thumb"],
                             "file_name": intake_db.file_name(e.title, e.file_id, intake_db.clean_date(e.date))}
                            for k, (_, e) in enumerate(edited.iterrows())])
            chosen = edited[edited["approve"]]
            if st.button(f"Approve {len(chosen)} selected", type="primary", disabled=chosen.empty):
                edits = {pid: (r.title, r.date, LABEL_TYPE.get(r.record_type, "")) for pid, r in chosen.iterrows()}
                n = intake_db.approve(b, edits, state.user, type_info)
                st.toast(f"{n} documents filed to the Archive")
                st.rerun()

    with st.container(key="zone-inreview"):
        st.markdown('<div class="rim-h2" role="heading" aria-level="2" style="margin-top:18px">Needs review</div>', unsafe_allow_html=True)
        if flagged.empty:
            st.caption("No document in this batch needs a person.")
        else:
            c1, c2 = st.columns([4, 1.4], vertical_alignment="center")
            c1.caption("Sent to the Review queue: confidence below the 55% cutoff, a weak reading, or a possible duplicate "
                       "(the file number is already filed).")
            if c2.button("Open the Review queue", use_container_width=True):
                state.tab = "Review"; st.rerun()
            fl = flagged.reset_index(drop=True)
            picked = st.dataframe(pd.DataFrame({
                "page": fl["thumb"].map(thumb_uri), "title": fl["title"],
                "why": fl["route_reason"], "department": fl["department_name"],
                "confidence": pd.to_numeric(fl["confidence"], errors="coerce"), "file_id": fl["file_id"],
                "status": fl["status"].map({"to review": "In the Review queue", "filed": "Filed from review"})}),
                hide_index=True, use_container_width=True, row_height=90,
                key=f"nr-{b}-{state.get('nr_nonce', 0)}", on_select="rerun", selection_mode="single-row",   # selecting a row opens the quick view
                column_config={"page": st.column_config.ImageColumn("Page", width="small"),
                               "confidence": st.column_config.ProgressColumn("Confidence", format="percent", min_value=0, max_value=1,
                                                                             color=BAR)})
            rows_sel = picked.selection.rows if picked and picked.selection else []
            if rows_sel and state.get("qv", {}).get("src") != "review":
                state.qv = {"src": "review", "i": rows_sel[0]}
            if state.get("qv", {}).get("src") == "review":
                quick_view([{"title": r.title, "file_id": r.file_id, "image": r.image, "thumb": r.thumb,
                             "file_name": r.file_name} for r in fl.itertuples()])
    if len(approved):
        with st.expander(f"Filed to the Archive from this batch · {len(approved)}"):
            st.dataframe(approved[["title", "date_shown", "department_name", "record_type_name", "file_name"]]
                         .rename(columns={"date_shown": "document date", "department_name": "department",
                                          "record_type_name": "record type", "file_name": "file name"}),
                         hide_index=True, use_container_width=True)
