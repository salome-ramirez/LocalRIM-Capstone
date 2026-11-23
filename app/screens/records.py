"""Archive screens: Archive home, Browse (folder tree and register with a preview) and the Reader."""
from __future__ import annotations

import html
import json
import re

import pandas as pd
import streamlit as st

from pipeline.common import ROOT
from .assistant import answer_lines, ask_panel
from .review import load_similar, plain_reason, similar_list
from .shared import APP, PER_PAGE, go, log, node_records, ocr_text, open_record, open_saved, read_history, read_saved, related, save_popover, search, show_image, write_saved

state = st.session_state
# Set by screens.use() on every run of app.py
archive = archive_raw = tax = type_info = review_queue = waiting_n = None


# Internal review codes (a prefix, a hyphen and a number) stay in archive.csv's how-filed text
# but are not shown in the Reader; an emptied pair of brackets goes too.
REVIEW_CODE = re.compile(r",?\s*\b(?:App|D|R)-\d+\b")


def shown_how(how: str) -> str:
    """Return the how-filed text for display, without review codes."""
    return re.sub(r"\s*\(\s*\)", "", REVIEW_CODE.sub("", how)).strip()


# The Reader opens in the view each user last chose, saved in outputs/app/reader_prefs.json.
READER_PREFS = APP / "reader_prefs.json"


READER_VIEWS = ["Page", "Text", "Side by side"]


def _read_prefs() -> dict:
    try:
        return json.loads(READER_PREFS.read_text(encoding="utf-8"))
    except Exception:
        return {}


def set_reader_view(mode: str):
    """Switch the Reader view and remember it for the current user."""
    state.mode = mode
    prefs = _read_prefs()
    if prefs.get(state.user) != mode:
        prefs[state.user] = mode
        READER_PREFS.write_text(json.dumps(prefs, indent=2, ensure_ascii=False), encoding="utf-8")


def register(view: pd.DataFrame, key: str):
    """Show a folder as a sortable table; a selected row opens a preview beside it."""
    view = view.reset_index(drop=True)
    ids_sig = abs(hash(tuple(view["id"]) + tuple(view.get("cross", pd.Series(dtype=str))))) % 10**8
    tkey = f"reg-{key}-{state.get('reg_nonce', 0)}-{ids_sig}"
    try:
        rows = st.session_state[tkey]["selection"]["rows"]
    except (KeyError, TypeError):
        rows = []
    rows = [i for i in rows if i < len(view)]

    def type_label(r):
        if r.get("cross", "") == "True":
            return f"Cross-reference · filed in {r['record_type_name'] or r['department_name']}"
        if r["record_type"] == "":
            return "No record type yet"
        return r["record_type_name"] + (" (suggested)" if r["suggested"] == "True" else "")

    table = pd.DataFrame({"date": view["date_iso"], "file_id": view["file_id"], "title": view["title"],
                          "type": view.apply(type_label, axis=1),
                          "pages": pd.to_numeric(view["pages"], errors="coerce"), "added": view["added"]})
    # fixed widths so the date, file ID and date added are never cut off
    cfg = {"date": st.column_config.TextColumn("Document date", width=112,
                                               help="As on the document: day, month or year only when that's all it shows"),
           "title": st.column_config.TextColumn("Title", width=290),
           "type": st.column_config.TextColumn("Record type", width=210),
           "file_id": st.column_config.TextColumn("File ID", width=128),
           "pages": st.column_config.NumberColumn("Pages", width=64),
           "added": st.column_config.TextColumn("Added", width=96)}
    open_ = bool(rows)
    if open_:
        table = table.drop(columns=["type", "added", "pages"])
        left, right = st.columns([2.2, 1.45], gap="medium")
    else:
        left, right = st.container(), None
    with left:
        st.dataframe(table, key=tkey, on_select="rerun", selection_mode="single-row", hide_index=True,
                     use_container_width=True, height=min(38 + 35 * len(table), 640),
                     column_config={k: v for k, v in cfg.items() if k in table.columns})
        c1, c2 = st.columns([3, 1.2], vertical_alignment="center")
        c1.caption("Click a record to preview it · click a column name to sort" if not open_
                   else "Click another record to preview it · Open full reader for every page and the record's history")
        c2.download_button("Export list (CSV)", view[["file_id", "title", "date_iso", "department_name", "series_name",
                                                      "record_type_name", "suggested", "pages", "file_name"]]
                           .rename(columns={"date_iso": "document_date", "department_name": "department",
                                            "series_name": "series", "record_type_name": "record_type",
                                            "suggested": "record_type_suggested"}).to_csv(index=False).encode(),
                           file_name=f"{key.replace('.', '-') or 'archive'}_list.csv", mime="text/csv",
                           key=f"exp-{tkey}", use_container_width=True)
    if not open_:
        return
    r = view.iloc[rows[0]]
    with right.container(border=True, key="rim-preview"):
        st.markdown(f'<div class="rim-h2" role="heading" aria-level="2" style="font-size:18px">{html.escape(r["title"])}</div>', unsafe_allow_html=True)
        img = ROOT / r["image"]
        if r["image"] and img.exists():
            show_image(img)
        tag = ' <span class="rim-tag">suggested</span>' if r["suggested"] == "True" else ""
        filed = " › ".join(x for x in (r["department_name"], r["series_name"], r["record_type_name"]) if x)
        also = f'<span>Also fits</span><span>{html.escape(r["also_fits_path"])}</span>' if r["also_fits_path"] else ""
        st.markdown(f'<div class="rim-grid"><span>Document date</span><span><b>{r["date_shown"]}</b></span>'
                    f'<span>File ID</span><span class="rim-mono">{r["file_id"]}</span>'
                    f'<span>Filed under</span><span>{html.escape(filed)}{tag}</span>{also}'
                    f'<span>Pages</span><span>{r["pages"]}</span></div>', unsafe_allow_html=True)
        if r.get("weak_reading", "") == "True":
            st.caption("Weak reading: the page has little readable text, so search may miss this record.")
        b1, b2 = st.columns(2)
        if b1.button("Open full reader", type="primary", key=f"open-{tkey}", use_container_width=True):
            open_record(r["id"], view["id"])
        pdf = ROOT / r["pdf"]
        if r["pdf"] and pdf.exists():
            b2.download_button("Download PDF", pdf.read_bytes(), file_name=r["file_name"] or pdf.name,
                               mime="application/pdf", key=f"dl-{tkey}", use_container_width=True)
        if st.button("Close preview", type="tertiary", key=f"close-{tkey}", use_container_width=True):
            state.reg_nonce = state.get("reg_nonce", 0) + 1
            st.rerun()


# Starter questions on Archive home; each was checked against the records it cites.
STARTERS = ["What did the records disposal notices ask employees to do?",
            "What did Philip Morris tell an outdoor advertising company after the Master Settlement Agreement?",
            "Which records were placed under disposal suspension, and why?",
            "List the safety data sheets in the archive, with each product name as written on the sheet."]


def home():
    """Archive home: question box and starters, department rail, recently filed records."""
    st.markdown('<div class="rim-sr" role="heading" aria-level="1">Archive home</div>', unsafe_allow_html=True)
    left, main = st.columns([1, 3.4], gap="large")
    # Left rail: record counts (the queue count opens the Review queue), then departments by size
    with left, st.container(key="rim-rail", gap="small"):
        st.markdown(f'<div class="rim-stat"><b>{len(archive)}</b><span>records in the Archive</span></div>', unsafe_allow_html=True)
        if st.button(f"**{len(review_queue)}** waiting for a person", type="tertiary", key="status-review"):
            state.tab = "Review"; st.rerun()
        st.markdown(f'<div class="rim-stat"><b>{int((archive["suggested"] == "True").sum())}</b><span>record types still suggested</span></div>',
                    unsafe_allow_html=True)
        st.markdown('<div class="rim-rail-h" role="heading" aria-level="2">Browse the files</div>', unsafe_allow_html=True)
        counts = [(f, int((archive["department"] == f["id"]).sum())) for f in tax["functions"]]
        for f, n in sorted([c for c in counts if c[1]], key=lambda c: -c[1]):
            with st.container(key=f"dept-{f['id']}", gap=None):
                row = st.container(key=f"deptrow-{f['id']}", horizontal=True, vertical_alignment="center", gap="small")
                if row.button(f["name"], key=f"home-{f['id']}", width="stretch"):
                    go("browse", node=("department", f["id"], f["name"]), shown=PER_PAGE)
                    st.rerun()
                row.markdown(f'<span class="rim-badge">{n}</span>', unsafe_allow_html=True, width="content")
                st.markdown(f'<div class="rim-share" aria-hidden="true"><i style="width:{100 * n / max(len(archive), 1):.0f}%"></i></div>',
                            unsafe_allow_html=True)
    with main:
        if waiting_n:
            w1, w2 = st.columns([4, 1.2], vertical_alignment="center")
            w1.markdown(f'<div class="rim-warn">{waiting_n} new document{"s" if waiting_n > 1 else ""} waiting for approval in Intake</div>',
                        unsafe_allow_html=True)
            if w2.button("Open Intake", use_container_width=True):
                state.tab = "Intake"; st.rerun()
        ask_panel("home", archive["id"].tolist(), "All records",
                  "Ask a question, or search a title, file ID or words on the page", starters=STARTERS)
        recent = node_records(("recent", "", "")).head(8)
        with st.container(key="zone-recent"), st.container(key="rim-list-recent", gap="small"):
            st.markdown('<div class="rim-card-h" role="heading" aria-level="2">Recently filed</div>', unsafe_allow_html=True)
            for r in recent.itertuples():
                c1, c2, c3 = st.columns([1.3, 4.6, 2.1], vertical_alignment="center")
                c1.markdown(f'<span class="rim-date">{html.escape(r.date_shown)}</span>', unsafe_allow_html=True)
                if c2.button(r.title, key=f"recent-{r.id}", use_container_width=True):
                    open_record(r.id, recent["id"])
                tag = ' <span class="rim-tag">suggested</span>' if r.suggested == "True" else ""
                c3.markdown(f'<span class="rim-idline"><span class="rim-muted rim-mono">{r.file_id}</span>{tag}</span>', unsafe_allow_html=True)


# Browse: folder tree and register with a preview
def browse():
    """Browse screen: folder tree, filters and the folder's register."""
    side, main = st.columns([1.05, 3.4], gap="large")
    with side, st.container(key="rim-tree", gap="small"):
        st.markdown('<div class="rim-eyebrow" role="heading" aria-level="2">Quick access</div>', unsafe_allow_html=True)
        quick = [("all", "", "All records", len(archive)),
                 ("recent", "", "Recently filed", 24),
                 ("untyped", "", "No record type yet", int((archive["record_type"] == "").sum()))]
        for level, key, label, n in quick:
            if st.button(f"{label} · {n}", key=f"q-{level}", use_container_width=True,
                         type="primary" if state.node[0] == level else "secondary"):
                go("browse", node=(level, key, label), shown=PER_PAGE)
                st.rerun()
        saved = read_saved()
        if saved:
            st.markdown('<div class="rim-eyebrow" role="heading" aria-level="2" style="margin-top:14px">Saved searches</div>', unsafe_allow_html=True)
            for item in saved:
                b1, b2 = st.columns([4, 1.6], vertical_alignment="center")
                if b1.button(item["name"], key=f"saved-{item['sid']}", use_container_width=True,
                             help=item["question"] if item["kind"] == "question" else None):
                    open_saved(item)
                if b2.button("Remove", key=f"unsave-{item['sid']}", type="tertiary"):
                    write_saved([x for x in saved if x["sid"] != item["sid"]]); st.rerun()
        st.markdown('<div class="rim-eyebrow" role="heading" aria-level="2" style="margin-top:14px">Departments</div>', unsafe_allow_html=True)
        for f in tax["functions"]:
            d = archive[archive["department"] == f["id"]]
            if d.empty:
                continue
            open_now = state.node[1].split(".")[0] == f["id"]
            with st.expander(f"{f['name']} · {len(d)}", expanded=open_now):
                for s in f["series"]:
                    ds = d[d["series"] == s["id"]]
                    if ds.empty:
                        continue
                    if st.button(f"{s['name']} · {len(ds)}", key=f"s-{s['id']}", use_container_width=True,
                                 type="primary" if state.node[1] == s["id"] else "secondary"):
                        go("browse", node=("series", s["id"], f"{f['name']} › {s['name']}"), shown=PER_PAGE)
                        st.rerun()
                    for t in s["types"]:
                        n = (ds["record_type"] == t["id"]).sum()
                        if n and st.button(f"{t['name']} · {n}", key=f"t-{t['id']}", use_container_width=True,
                                           type="primary" if state.node[1] == t["id"] else "secondary"):
                            go("browse", node=("type", t["id"], f"{f['name']} › {s['name']} › {t['name']}"), shown=PER_PAGE)
                            st.rerun()

    with main:
        level, key, label = state.node
        view = node_records(state.node)
        crumbs, title = label.rsplit(" › ", 1) if " › " in label else ("Archive", label)
        dated = view[view["date_iso"] != "undated"]["date_iso"]
        span = f" · {dated.min()[:4]}–{dated.max()[:4]}" if len(dated) else ""
        _brbar = st.container(key="zone-brbar")
        _brbar.markdown(f'<div class="rim-muted">{html.escape(crumbs)}</div><div class="rim-h1" role="heading" aria-level="1">{html.escape(title)}</div>'
                        f'<div class="rim-muted">{len(view)} records{span}</div>', unsafe_allow_html=True)

        if state.get(f"dec-{key}") is not None and state[f"dec-{key}"] not in (["All"] + view["decade"].unique().tolist()):
            state[f"dec-{key}"] = "All"                         # a saved date filter that no longer applies
        # one row that wraps when narrow, so no field or label is cut
        c1 = c2 = c3 = c4 = _brbar.container(key=f"fbar-{key}", horizontal=True, gap="small", vertical_alignment="bottom")
        within = c1.text_input("Search in this folder", placeholder="Title, file ID or words on the page", key=f"in-{key}")
        decades = ["All"] + sorted(d for d in view["decade"].unique() if d != "Undated") + (["Undated"] if (view["decade"] == "Undated").any() else [])
        counts = {d: (view["decade"] == d).sum() for d in decades}
        decade = c2.selectbox("Document date", decades, width=180, format_func=lambda d: f"{d} · {len(view) if d == 'All' else counts[d]}",
                              key=f"dec-{key}")
        order = c3.selectbox("Sort", ["Oldest first", "Newest first"], key=f"ord-{key}", width=160)
        what = within.strip() or (decade if decade != "All" else "")
        save_popover(c4, f"{title} · {what}", {"kind": "folder", "node": list(state.node), "within": within.strip(),
                                                "decade": decade, "order": order}, f"folder-{key}",
                     disabled=not within.strip() and decade == "All", width="content")
        if within.strip():
            hits = set(search(within)["id"])
            view = view[view["id"].isin(hits)]
        if decade != "All":
            view = view[view["decade"] == decade]
        undated = view[view["date_iso"] == "undated"]
        view = pd.concat([view[view["date_iso"] != "undated"].sort_values("date_iso", ascending=order == "Oldest first"), undated])
        if view.empty:
            st.markdown('<div class="rim-empty">No records here match. Try another folder or clear the search.</div>', unsafe_allow_html=True)
        open_ask = f"start-folder-{key}" in state or f"ans-folder-{key}" in state
        with st.expander(f"Ask about the {len(view)} records in this folder", expanded=open_ask):
            ask_panel(f"folder-{key}", view["id"].tolist(), title, "e.g. Which of these mention billboards?")
        register(view, key or level)


# Reader
@st.cache_data
def pdf_page_png(pdf_path: str, page: int) -> bytes | None:
    """Render one page of a PDF as PNG bytes, or None if it can't be read."""
    try:
        import fitz
        with fitz.open(ROOT / pdf_path) as doc:
            return doc[page - 1].get_pixmap(dpi=110).tobytes("png")
    except Exception:
        return None


def reader():
    """Reader screen: the page or its text, the record's details, history and related records."""
    r = archive[archive["id"] == state.record].iloc[0]
    ids = state.list or [r["id"]]
    pos = ids.index(r["id"]) if r["id"] in ids else 0
    folder = state.node[2]

    # Top bar: back link, previous/next record, view switch
    top = st.container(key="zone-rdrbar").columns([2, 2, 2], vertical_alignment="center")
    with top[0].container(key="rdr-back"):
        if st.button(f"‹ {folder}", type="tertiary", key="back"):
            go("browse")
            st.rerun()
    with top[1].container(key="rdr-nav", horizontal=True, vertical_alignment="center", gap="small"):
        if st.button("‹", disabled=pos == 0, key="prev"):
            go("reader", record=ids[pos - 1], pdf_page=1); st.rerun()
        st.markdown(f'<div class="rim-muted">Record {pos + 1} of {len(ids)}</div>', unsafe_allow_html=True)
        if st.button("›", disabled=pos >= len(ids) - 1, key="next"):
            go("reader", record=ids[pos + 1], pdf_page=1); st.rerun()
    with top[2].container(key="rdr-view", horizontal=True):
        chosen = st.segmented_control("View", READER_VIEWS, default=state.mode, label_visibility="collapsed")
    if chosen and chosen != state.mode:
        set_reader_view(chosen)

    if state.mode == "Side by side":          # page and text share the full width; details move below
        doc_col, panel = st.container(), st.container()
    else:
        doc_col, panel = st.columns([2.3, 1.15], gap="large")
    with doc_col, st.container(key="zone-viewer-rd"):
        pages = int(r["pages"] or 1)
        text = ocr_text(r["text"])
        hl = state.get("hl")
        marks = answer_lines(text, hl["sentences"]) if hl and hl["id"] == r["id"] and text else None

        def page_view():
            if marks and state.mode == "Page":
                b1, b2 = st.columns([3, 2], vertical_alignment="center")
                b1.markdown('<div class="rim-hlbar"><b>Cited in the answer</b></div>', unsafe_allow_html=True)
                if b2.button(f"Show the {len(marks)} matching line{'s' if len(marks) != 1 else ''}", key="hl-show",
                             use_container_width=True):
                    set_reader_view("Text"); st.rerun()
            if state.pdf_page == 1 and (ROOT / r["image"]).exists():
                show_image(ROOT / r["image"])
                st.markdown(f'<div class="rim-sr">Scanned first page of \'{html.escape(r["title"])}\'. '
                            f'Choose Text above for the words on this page.</div>', unsafe_allow_html=True)
            else:
                png = pdf_page_png(r["pdf"], state.pdf_page)
                if png:
                    show_image(png)
                else:
                    st.markdown('<div class="rim-empty">This page can\'t be shown. Download the PDF to read it.</div>', unsafe_allow_html=True)
            if pages > 1:
                p1, p2, p3 = st.columns([1, 2, 1], vertical_alignment="center")
                if p1.button("‹ Page", disabled=state.pdf_page == 1):
                    state.pdf_page -= 1; st.rerun()
                p2.markdown(f'<div class="rim-muted" style="text-align:center">Page {state.pdf_page} of {pages} · filed from page 1</div>', unsafe_allow_html=True)
                if p3.button("Page ›", disabled=state.pdf_page == pages):
                    state.pdf_page += 1; st.rerun()

        def text_view():
            if marks is not None:
                b1, b2 = st.columns([3, 1.3], vertical_alignment="center")
                if marks:
                    links = " · ".join(f'<a href="#hl-{k}">{k}</a>' for k in range(1, len(marks) + 1))
                    count = f"{len(marks)} line{'s' if len(marks) != 1 else ''} match{'es' if len(marks) == 1 else ''} the answer"
                    b1.markdown(f'<div class="rim-hlbar"><b>{count}</b> · {links}</div>', unsafe_allow_html=True)
                else:
                    b1.markdown('<div class="rim-hlbar">No line in this record shares the answer\'s key words or numbers; '
                                'the answer may draw on the page as a whole.</div>', unsafe_allow_html=True)
                if b2.button("Clear highlights", key="hl-clear", use_container_width=True):
                    state.pop("hl", None); st.rerun()
                if marks:
                    st.caption("Matched by shared words and numbers, not by meaning. Read each line in context.")
            find = st.text_input("Find in this document", key=f"find-{r['id']}")
            if text:
                f = html.escape(find.strip()) if find.strip() else ""
                order = {i: k for k, i in enumerate(marks or [], 1)}
                lines = []
                for i, line in enumerate(text.split("\n")):
                    esc = html.escape(line)
                    if f:
                        esc = esc.replace(f, f"<mark>{f}</mark>")
                    if i in order:
                        esc = f'<span class="rim-hl" id="hl-{order[i]}">{esc}</span>'
                    lines.append(esc)
                body = "\n".join(lines)
            else:
                body = "No text was read from this page."
            st.markdown(f'<div class="rim-text">{body}</div>', unsafe_allow_html=True)
            st.caption(f"Text read from page 1 (the page used for filing) · {r['ocr_words']} words · "
                       f"median word confidence {float(r['ocr_conf']):.0f}")

        if state.mode == "Page":
            page_view()
        elif state.mode == "Text":
            text_view()
        else:
            a, b = st.columns(2)
            with a:
                page_view()
            with b:
                text_view()

    with panel:
      # Record card (Details, History, Related), record type card, then Ask
      with st.container(key="zone-rdrinfo"):
        details, history, rel = st.tabs(["Details", "History", "Related"])
        with details:
            st.markdown(f'<div class="rim-h2" role="heading" aria-level="1">{html.escape(r["title"])}</div>'
                        f'<div class="rim-grid"><span>Document date</span><span><b>{r["date_shown"]}</b></span>'
                        f'<span>File ID</span><span class="rim-mono">{r["file_id"]}</span>'
                        f'<span>Department</span><span>{html.escape(r["department_name"])}</span>'
                        + (f'<span>Also fits</span><span>{html.escape(r["also_fits_path"])}</span>' if r["also_fits_path"] else "")
                        + f'<span>Pages</span><span>{r["pages"]}</span></div>', unsafe_allow_html=True)
            if r["weak_reading"] == "True":
                st.markdown('<div class="rim-warn" style="margin-top:10px">Text reading weak: search and the assistant may miss this record.</div>',
                            unsafe_allow_html=True)
            st.markdown(f'<div class="rim-file">{html.escape(r["file_name"])}</div>', unsafe_allow_html=True)
            pdf = ROOT / r["pdf"]
            if pdf.exists():
                st.download_button(f"Download PDF · {r['pages']} page{'s' if r['pages'] != '1' else ''}", pdf.read_bytes(),
                                   file_name=r["file_name"], mime="application/pdf", use_container_width=True)
        with history:
            for h in read_history()[lambda d: d["id"] == r["id"]].iloc[::-1].itertuples():
                if h.action == "confirmed":
                    st.markdown(f"**{h.time}** · record type confirmed by {h.by}")
                else:
                    old = type_info.get(h.old, ("", "", "", "", h.old or "none"))[4]
                    new = type_info.get(h.new, ("", "", "", "", h.new))[4]
                    st.markdown(f"**{h.time}** · filing changed by {h.by}: {old} → {new}" + (f" · *{h.note}*" if h.note else ""))
            st.markdown(f"**{r['added']}** · {shown_how(r['how_filed'])}")
            if (r["suggested"] == "True" or "pipeline" in r["how_filed"]) and r["id"] in set(archive_raw["id"]):
                st.caption(f"Original filing kept for audit: {archive_raw.set_index('id').loc[r['id'], 'department_name']} › "
                           f"{archive_raw.set_index('id').loc[r['id'], 'record_type_name'] or 'no record type'}")
        with rel:
            rel_df = related(r["id"])
            for _, x in rel_df.iterrows():
                c1, c2 = st.columns([4, 1], vertical_alignment="center")
                c1.markdown(f'<div class="rim-date">{x["date_shown"]}</div><div>{html.escape(x["title"])}</div>'
                            f'<div class="rim-muted">{html.escape(x["department_name"])} · <span class="rim-mono">{x["file_id"]}</span></div>',
                            unsafe_allow_html=True)
                if c2.button("Open", key=f"rel-{x['id']}"):
                    open_record(x["id"], rel_df["id"])
            st.caption("Records whose pages read most alike (the same search the review helper uses).")
      with st.container(key="zone-rdrtype"):
        tag = '<span class="rim-tag">suggested</span>' if r["suggested"] == "True" else ""
        path = f'{r["series_name"]} › {r["record_type_name"]}' if r["record_type"] else "No record type yet"
        st.markdown(f'<div class="rim-card-h" role="heading" aria-level="2">Record type {tag}</div><div style="font-weight:600;margin:4px 0">{html.escape(path)}</div>',
                    unsafe_allow_html=True)
        if r["suggested"] == "True" and r["reason"]:
            st.caption(f"Why: {plain_reason(r['reason'], r['id'] in load_similar())}")
            similar_list(r["id"], f"rd-{r['id']}")
        b1 = b2 = st.container(horizontal=True, gap="small")   # wraps when narrow, so "Change filing" is never cut
        if r["suggested"] == "True" and b1.button("Confirm", type="primary", width="stretch"):
            log(r["id"], "confirmed", r["record_type"], r["record_type"])
            st.rerun()
        with b2.popover("Change filing", width="content"):
            options = sorted(type_info, key=lambda t: (type_info[t][1], type_info[t][3], type_info[t][4]))
            new = st.selectbox("File as", options, index=options.index(r["record_type"]) if r["record_type"] in options else 0,
                               format_func=lambda t: f"{type_info[t][1]} › {type_info[t][3]} › {type_info[t][4]}")
            note = st.text_input("Reason (optional)")
            if st.button("Save filing", type="primary"):
                log(r["id"], "changed", r["record_type"], new, note)
                st.rerun()
      ask_panel(f"rec-{r['id']}", [r["id"]], "this record", "e.g. What does it cost per month?")
