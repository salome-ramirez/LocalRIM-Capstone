"""Review queue: one page at a time with the review helper's suggestion."""
from __future__ import annotations

import html
import re
from pathlib import Path

import pandas as pd
import streamlit as st

from pipeline.common import ROOT
from localrim import intake as intake_db
from localrim.intake import REASONS
from .shared import ocr_text, open_record, show_image

state = st.session_state
# Set by screens.use() on every run of app.py
archive = tax = dept_name = type_info = review_queue = TYPE_OPTIONS = TYPE_LABEL = CODE_NAME = _CODES = None


# The helper's reasons in plain words, and the five similar filed pages the model saw (S1-S5,
# rank 1-5 in results/scale_run/neighbors.csv). Pages added through Add files use new_neighbors.csv.
@st.cache_data
def _similar_from(files: tuple) -> dict:
    parts = [pd.read_csv(f, dtype={"id": str, "neighbor_id": str}) for f, _ in files]
    if not parts:
        return {}
    nb = pd.concat(parts, ignore_index=True)
    nb["rank"] = pd.to_numeric(nb["rank"], errors="coerce")
    nb = nb.sort_values(["id", "rank"])
    return {i: g["neighbor_id"].tolist() for i, g in nb.groupby("id")}


def load_similar() -> dict:
    """Return record ID -> its five similar filed pages (S1-S5), from the scale run and uploads.

    Re-read when either file changes.
    """
    files = [ROOT / "results" / "scale_run" / "neighbors.csv", Path(intake_db.NEW_VEC_IDS).parent / "new_neighbors.csv"]
    return _similar_from(tuple((str(f), f.stat().st_mtime) for f in files if f.exists()))


_SREF = r"S\d(?:(?:, and |, | and |/)S\d)*"


def _low(name: str) -> str:
    return name[0].lower() + name[1:] if len(name) > 1 and name[1].islower() else name


def _pages(ref: str) -> str:
    n = re.findall(r"S(\d)", ref)
    return ("filed page " + n[0]) if len(n) == 1 else "filed pages " + ", ".join(n[:-1]) + " and " + n[-1]


def plain_reason(text: str, has_list: bool) -> str:
    """Rewrite the helper's reason for display.

    Codes become names; S-references become numbered filed pages (only when the list they
    point to can be shown); (R9) becomes (rule R9).
    """
    if not text:
        return text
    t = text
    # A: "name (CODE)" -> "name";  otherwise "(CODE)" -> "(name)"
    t = re.sub(rf"([^()]*?)\s*\(({_CODES})\)",
               lambda m: m.group(1) if CODE_NAME[m.group(2)].lower() in m.group(1).lower()[-len(CODE_NAME[m.group(2)]) - 30:]
               else f"{m.group(1)} ({_low(CODE_NAME[m.group(2)])})", t)
    # B: "CODE (name)" -> "name"
    t = re.sub(rf"\b({_CODES})\s*\(([^)]*)\)",
               lambda m: _low(m.group(2)) if m.group(2).lower().startswith(CODE_NAME[m.group(1)].lower()[:8])
               else f"{_low(CODE_NAME[m.group(1)])} ({m.group(2)})", t)
    # C: a bare code -> its name
    t = re.sub(rf"\b({_CODES})\b(?!\w|\.\w)", lambda m: _low(CODE_NAME[m.group(1)]), t)
    # D: similar pages -> "filed page n" / "similar filed page n"
    if has_list:
        t = re.sub(rf"\bsimilar (?:filed )?(?:pages?|documents?|examples?) \(?({_SREF})\)?",
                   lambda m: "similar " + _pages(m.group(1)), t, flags=re.I)
        t = re.sub(rf"\(({_SREF})\)", lambda m: "(" + _pages(m.group(1)) + ")", t)
        t = re.sub(rf"\b({_SREF})\b", lambda m: _pages(m.group(1)), t)
    # E: rules
    t = re.sub(r"\((R\d+)\)", r"(rule \1)", t)
    return t


def similar_list(rid: str, key: str):
    """Show the similar filed pages, numbered as the model saw them, each opening in the Reader.

    A hidden page keeps its number but has no title or link.
    """
    ids = load_similar().get(rid)
    if not ids:
        return
    by_id = archive.set_index("id")
    st.markdown('<div class="rim-eyebrow" role="heading" aria-level="3" style="margin-top:6px">Similar filed pages</div>', unsafe_allow_html=True)
    with st.container(key=f"simlist-{key}", gap=None):
        shown = [i for i in ids if i in by_id.index and i not in intake_db.HIDDEN]
        for n, nid in enumerate(ids, 1):
            if nid in intake_db.HIDDEN or nid not in by_id.index:
                st.markdown(f'<div class="rim-sim-x">{n}. Not shown (withheld for privacy)</div>', unsafe_allow_html=True)
                continue
            x = by_id.loc[nid]
            kind = x["record_type_name"] or x["department_name"]
            label = f"{n}. {x['title']}" + (f" · {kind}" if kind and kind.lower() != str(x['title']).lower() else "")
            if st.button(label, key=f"sim-{key}-{n}", type="tertiary"):
                state.tab = "Archive"
                open_record(nid, shown)


def based_on(sources: str) -> str:
    """Summarise a sources string: 'S1; S3; CLASSIFIER; R1' -> '2 similar filed pages, the classifier's guess, rule R1'."""
    toks = [t.strip() for t in re.split(r"[;,]", sources or "") if t.strip()]
    s_n = sum(1 for t in toks if re.fullmatch(r"S\d+", t))
    rules = [t for t in toks if re.fullmatch(r"R\d+", t)]
    parts = []
    if s_n:
        parts.append(f"{s_n} similar filed page{'s' if s_n != 1 else ''}")
    if "CLASSIFIER" in toks:
        parts.append("the classifier's guess")
    if rules:
        parts.append("rule " + rules[0] if len(rules) == 1 else "rules " + ", ".join(rules[:-1]) + " and " + rules[-1])
    return ", ".join(parts) or sources


def review_screen():
    """Review queue screen: one page at a time with the helper's suggestion and the filing form."""
    q = review_queue
    _rqbar = st.container(key="zone-rqbar")
    _rqbar.markdown(f'<div class="rim-eyebrow">Review queue</div><div class="rim-h1" role="heading" aria-level="1">{len(q)} pages need a person</div>',
                    unsafe_allow_html=True)
    if q.empty:
        st.markdown('<div class="rim-empty">Nothing waiting. Every page has been filed.</div>', unsafe_allow_html=True)
        return
    f1, f2 = _rqbar.columns([3, 2], vertical_alignment="bottom")
    show = f2.segmented_control("Show", ["All", "New uploads", "Second look"], default=state.get("rq_filter", "All"),
                                key="rq_show") or "All"
    if show != state.get("rq_filter", "All"):
        state.rq_filter, state.rq = show, 0
    if show == "New uploads":
        q = q[q["from"] != "scale run"].reset_index(drop=True)
    elif show == "Second look":
        q = q[q["second_look"]].reset_index(drop=True)
    if q.empty:
        st.markdown('<div class="rim-empty">No pages in this view.</div>', unsafe_allow_html=True)
        return
    state.rq = min(state.get("rq", 0), len(q) - 1)
    r = q.iloc[state.rq]

    nav = f1.columns([0.6, 1.6, 0.6, 3], vertical_alignment="center")
    if nav[0].button("‹", disabled=state.rq == 0, key="rq-prev"):
        state.rq -= 1; st.rerun()
    nav[1].markdown(f'<div class="rim-muted" style="text-align:center">Page {state.rq + 1} of {len(q)}</div>', unsafe_allow_html=True)
    if nav[2].button("›", disabled=state.rq >= len(q) - 1, key="rq-next"):
        state.rq += 1; st.rerun()

    page, panel = st.columns([2.1, 1.3], gap="large")
    with page, st.container(key="zone-viewer-rq"):
        mode = st.segmented_control("View", ["Page", "Text"], default="Page", key=f"rq-mode-{r['id']}",
                                    label_visibility="collapsed") or "Page"
        if mode == "Page" and r["image"] and (ROOT / r["image"]).exists():
            show_image(ROOT / r["image"])
            st.markdown(f'<div class="rim-sr">Scanned page waiting for review: \'{html.escape(r["title"])}\'. '
                        f'Choose Text above for the words on this page.</div>', unsafe_allow_html=True)
        else:
            text = ocr_text(r["text"]) if r["text"] else ""
            st.markdown(f'<div class="rim-text">{html.escape(text) if text else "No text was read from this page."}</div>',
                        unsafe_allow_html=True)

    with panel.container(key="rq-panel", gap="small"):
        why = REASONS.get(r["route_reason"], r["route_reason"]) or "Needs review"
        source = (f"from batch {r['batch']} · uploaded by {html.escape(intake_db.read(intake_db.BATCHES).set_index('batch').loc[r['batch'], 'uploaded_by'])}"
                  if r["from"] != "scale run" else "from the scale run")
        if r.get("duplicate_of"):
            st.caption(f"File number {r['duplicate_of']} is already filed. Check it isn't the same document.")
        conf = pd.to_numeric(r["confidence"], errors="coerce")
        conf2 = pd.to_numeric(r["second_confidence"], errors="coerce")
        sug = r.get("suggested_record_type", "") or r.get("record_type", "")
        reason = plain_reason((r["reason"] or "").strip(), r["id"] in load_similar())
        # the suggestion and its reason first, so the reviewer reads why before acting
        with st.container(border=True, key="rq-suggest", gap="small"):
            st.markdown('<div class="rim-card-h" role="heading" aria-level="2">Review helper suggests</div>', unsafe_allow_html=True)
            if sug in type_info:
                st.markdown(f'<p class="rq-type">{html.escape(TYPE_LABEL[sug])}</p>', unsafe_allow_html=True)
                if reason:
                    st.markdown(f'<p class="rq-reason">{html.escape(reason)}</p>', unsafe_allow_html=True)
            else:
                st.caption("No suggestion for this page (the assistant was unavailable or the page has no readable text).")
            # why the page is with a person
            st.markdown(f'<p class="rq-why"><b>Needs review · {html.escape(why)}</b> · {source}'
                        + (" · <b>second look requested</b>" if r["second_look"] else "") + "</p>", unsafe_allow_html=True)

        with st.container(key="zone-rqfile", gap="small"):
            st.markdown('<div class="rim-card-h" role="heading" aria-level="2">File it</div>', unsafe_allow_html=True)
            pid = r["id"]
            options = ["(department only, no record type)"] + TYPE_OPTIONS
            t = st.selectbox("Record type", options, index=options.index(sug) if sug in type_info else 0,
                             format_func=lambda x: TYPE_LABEL.get(x, x), key=f"rq-type-{pid}")
            dept = r["classifier_department"]
            if t not in type_info:
                depts = [f["id"] for f in tax["functions"]]
                dept = st.selectbox("Department", depts, index=depts.index(dept) if dept in depts else 0,
                                    format_func=lambda d: dept_name[d], key=f"rq-dept-{pid}")
            td = st.container(key="rq-td", horizontal=True, gap="small")   # title and date side by side; the date wraps under when narrow
            title = td.text_input("Title", value=r["title"], key=f"rq-title-{pid}")
            date = td.text_input("Document date", value=r["date_iso"], key=f"rq-date-{pid}", width=150,
                                 help="YYYY-MM-DD, YYYY-MM, YYYY or undated")
            iso = intake_db.clean_date(date)
            st.markdown(f'<div class="rim-file">{html.escape(intake_db.file_name(title, r["file_id"], iso))}</div>', unsafe_allow_html=True)
            if date.strip() and iso == "undated" and date.strip().lower() != "undated":
                st.caption("That date isn't in YYYY-MM-DD, YYYY-MM or YYYY form, so it will be saved as undated.")
            if st.button("File to Archive and next", type="primary", use_container_width=True, key=f"rq-file-{pid}"):
                intake_db.file_from_review(r.to_dict(), title, iso, t, dept, state.user, type_info, dept_name)
                st.toast(f"Filed: {title}")
                st.rerun()
            acts = st.container(horizontal=True, gap="small")
            b1 = b2 = acts
            if b1.button("Skip for now", key=f"rq-skip-{pid}"):
                state.rq = (state.rq + 1) % len(q); st.rerun()
            if not r["second_look"] and b2.button("Ask a colleague for a second look", key=f"rq-2nd-{pid}"):
                intake_db.review_action(pid, "second look requested", state.user)
                st.toast("Marked for a second look")
                st.rerun()

        # the classifier's guess and the evidence go below the actions
        if pd.notna(conf) or (sug in type_info and (reason or r["sources"])):
            with st.expander("Evidence"):
                if pd.notna(conf):
                    st.markdown(f'<div class="rim-muted" style="margin:4px 0 8px">The classifier&#39;s guess: <b>{html.escape(dept_name.get(r["classifier_department"], r["classifier_department"]))}</b>, '
                                f'{conf:.0%} · then {html.escape(dept_name.get(r["second"], r["second"]))}, {conf2:.0%}</div>', unsafe_allow_html=True)
                if sug in type_info:
                    if r["sources"]:
                        st.caption(f"Based on: {based_on(r['sources'])}")
                similar_list(r["id"], f"rq-{r['id']}")
