"""Archive assistant: search by meaning and words, answer with sources, highlight the lines
an answer draws on."""
from __future__ import annotations

import html
import re
from datetime import datetime

import pandas as pd
import streamlit as st

from pipeline.common import ROOT
from localrim.assistant import LLM, ask_llm, cited, retrieve as _retrieve
from .shared import open_record, related, save_popover

state = st.session_state
# Set by screens.use() on every run of app.py
archive = TYPE_WORDS = None


NO_OLLAMA = "Assistant unavailable without Ollama. Showing only the most relevant documents for now."


def retrieve(q: str, ids: list[str]) -> pd.DataFrame:
    """Search the archive for q within ids."""
    return _retrieve(q, archive, ids, TYPE_WORDS)


# Highlighting: a line is marked when it shares enough key words or numbers with the answer.
# A shared word or short number counts 1; a number of 3+ digits or a full date counts 2. A line needs 2.
HL_STOP = set("""about above after again against also among another because been before being below between both
could does doing down during each few from further have having here into itself just more most much must only other
over same should some such than that their them then there these they this those through under until very were what
when where which while will with within without would your yours records record document documents page pages
answer archive according shows show stated states appears seems seem""".split())


HL_MAX = 8


MONTHS = {m: i for i, m in enumerate(["january", "february", "march", "april", "may", "june", "july", "august",
                                      "september", "october", "november", "december"], 1)}


def _full_dates(text: str) -> set:
    """Full dates (year, month, day) written as "January 1, 1996", "1 January 1996", "02/11/98" or "1996-01-02"."""
    out = set()
    for mo, d, y in re.findall(r"\b([a-z]{3,9})\.? (\d{1,2}),? (\d{4})\b", text):
        if mo in MONTHS or mo[:3] in {m[:3] for m in MONTHS}:
            out.add((int(y), next(v for k, v in MONTHS.items() if k.startswith(mo[:3])), int(d)))
    for d, mo, y in re.findall(r"\b(\d{1,2}) ([a-z]{3,9})\.?,? (\d{4})\b", text):
        if mo[:3] in {m[:3] for m in MONTHS}:
            out.add((int(y), next(v for k, v in MONTHS.items() if k.startswith(mo[:3])), int(d)))
    for mo, d, y in re.findall(r"\b(\d{1,2})/(\d{1,2})/(\d{2}|\d{4})\b", text):
        y = int(y); y = y + 1900 if y < 100 else y
        out.add((y, int(mo), int(d)))
    for y, mo, d in re.findall(r"\b(\d{4})-(\d{2})-(\d{2})\b", text):
        out.add((int(y), int(mo), int(d)))
    return out


def _hl_terms(text: str):
    text = re.sub(r"\[\d+\]", " ", text.lower())
    words = {w.rstrip("s") for w in re.findall(r"[a-z]{4,}", text)} - {w.rstrip("s") for w in HL_STOP}
    nums = {n.replace(",", "") for n in re.findall(r"\d[\d,]*", text)}
    return words, nums, _full_dates(text)


def answer_lines(text: str, sentences: list[str]) -> list[int]:
    """Return the indexes of the OCR lines to highlight, in reading order.

    A shared word or a one- or two-digit number counts 1; a number of three or more digits
    (a year, a code) or a full date counts 2. A line needs 2. At most HL_MAX lines, the best
    matching ones.
    """
    words, nums, dates = _hl_terms(" ".join(sentences))
    scored = []
    for i, line in enumerate(text.split("\n")):
        low = line.lower()
        lw = {w.rstrip("s") for w in re.findall(r"[a-z]{4,}", low)}
        shared = nums & {n.replace(",", "") for n in re.findall(r"\d[\d,]*", low)}
        score = (len(words & lw) + sum(2 if len(n) >= 3 else 1 for n in shared)
                 + 2 * len(dates & _full_dates(low)))
        if score >= 2:
            scored.append((score, i))
    best = sorted(scored, key=lambda t: (-t[0], t[1]))[:HL_MAX]
    return sorted(i for _, i in best)


def related_to(ids: list[str], exclude: set, k: int = 6) -> pd.DataFrame:
    """Return records similar in meaning to any of ids, merged rank by rank, excluding exclude."""
    lists = [related(i, k + len(exclude))["id"].tolist() for i in ids]
    out = []
    for rank in range(max((len(l) for l in lists), default=0)):
        for l in lists:
            if rank < len(l) and l[rank] not in exclude and l[rank] not in out:
                out.append(l[rank])
    out = out[:k]
    return archive.set_index("id").loc[out].reset_index() if out else archive.iloc[0:0]


def answer_html(answer: str, disp: dict) -> str:
    """Return the answer as HTML, with source numbers in citation order and listed items as a list."""
    text = html.escape(answer)
    text = re.sub(r"\[(\d+)\]", lambda m: f"[{disp.get(int(m.group(1)), m.group(1))}]", text)
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    if sum(l.startswith("- ") for l in lines) >= 2:                      # one item per line
        head = [l for l in lines if not l.startswith("- ")]
        items = [l[2:] for l in lines if l.startswith("- ")]
        return "".join(f"<p style='margin:0'>{h}</p>" for h in head) + "<ul>" + "".join(f"<li>{i}</li>" for i in items) + "</ul>"
    m = re.search(r":\s+-\s+", text)
    if m:                                                               # items on one line, each ending with its source
        items = re.split(r"(?<=\])\s+-\s+", text[m.end():])
        if len(items) >= 2:
            return f"<p style='margin:0'>{text[:m.start() + 1]}</p><ul>" + "".join(f"<li>{i}</li>" for i in items) + "</ul>"
    return text


def citing_sentences(answer: str, n: int) -> list[str]:
    """Return the sentences of the answer that cite source n."""
    parts = re.split(r"(?<=[.!?])\s+", answer or "")
    return [p for p in parts if f"[{n}]" in p] or [answer or ""]


def bundle_pdf(docs: pd.DataFrame) -> bytes | None:
    """Return the records' PDFs merged into one, or None if there are none."""
    try:
        import fitz
        out = fitz.open()
        for d in docs.itertuples():
            p = ROOT / d.pdf
            if p.exists():
                with fitz.open(p) as src:
                    out.insert_pdf(src)
        return out.tobytes() if out.page_count else None
    except Exception:
        return None


def sources_csv(q: str, docs: pd.DataFrame) -> bytes:
    """Return the source list of an answer as CSV bytes."""
    rows = pd.DataFrame({
        "source": docs["n"] if "n" in docs else range(1, len(docs) + 1), "file_id": docs["file_id"], "title": docs["title"],
        "document_date": docs["date_iso"], "department": docs["department_name"],
        "series": docs["series_name"], "record_type": docs["record_type_name"],
        "record_type_status": docs["suggested"].map({"True": "suggested"}).fillna("final"),
        "file_name": docs["file_name"], "question": q, "asked_by": state.user,
        "asked_at": datetime.now().strftime("%Y-%m-%d %H:%M")})
    return rows.to_csv(index=False).encode("utf-8")


def ask_panel(key: str, ids: list[str], scope_label: str, placeholder: str, starters: list[str] | None = None):
    """Show the question box, the answer and its sources.

    The last answer per box is kept in session state. Follow-up actions can narrow the scope
    to one department or to the cited sources, or list related records; a narrowed scope is
    kept in state[f"scope-{key}"] until the user goes back.
    """
    home_label = scope_label
    back_label = "Back to all records" if home_label == "All records" else f"Back to {home_label}"
    scope = state.get(f"scope-{key}")
    if scope:
        ids, scope_label = scope["ids"], scope["label"]
        if scope["kind"] == "sources":
            if st.button(back_label, key=f"back-{key}", type="tertiary"):
                state.pop(f"scope-{key}", None); state.pop(f"rel-{key}", None); st.rerun()

    def run(question):
        with st.spinner("Searching the archive and writing an answer. This usually takes about 20 seconds on this computer."):
            docs = retrieve(question, ids)
            answer = ask_llm(question, docs) if not docs.empty else "Nothing in this scope matches."
        state[f"ans-{key}"] = (question, answer, docs)
        state[f"ansn-{key}"] = state.get(f"ansn-{key}", 0) + 1   # a new key per answer, so the answer block animates in once
        state[f"announce-{key}"] = True                     # screen-reader announcement, once per answer
        state.pop(f"rel-{key}", None)

    pending = state.pop(f"start-{key}", None)              # a starter question was clicked
    if pending:
        state[f"q-{key}"] = pending                         # set before the box is drawn, so it shows the question
    with st.form(f"ask-{key}"):
        c1, c2 = st.columns([3, 1.2] if key.startswith("rec-") else [6, 1], vertical_alignment="bottom")
        q = c1.text_input(f"Search or ask · in {scope_label}", placeholder=placeholder, key=f"q-{key}")
        asked = c2.form_submit_button("Ask", type="secondary" if key.startswith("rec-") else "primary",
                                      use_container_width=True)
    if asked and q.strip():
        run(q)
    elif pending:
        run(pending)
    elif state.pop(f"rerun-{key}", None):
        run(state[f"ans-{key}"][0])
    # Starter questions, shown until the first answer
    if starters and f"ans-{key}" not in state:
        st.markdown('<div class="rim-muted" style="margin:2px 0 6px">Try one of these</div>', unsafe_allow_html=True)
        box = st.container(key=f"starters-{key}", gap=None)
        for i, question in enumerate(starters):
            if box.button(question, key=f"starter-{key}-{i}", type="tertiary", wrap=True):
                state[f"start-{key}"] = question; st.rerun()
    if f"ans-{key}" not in state:
        return
    q, answer, docs = state[f"ans-{key}"]
    if scope and scope["kind"] == "dept":
        n1, n2 = st.columns([3, 1], vertical_alignment="center")
        n1.markdown(f'<div class="rim-muted">Answer from {html.escape(scope["label"])} only</div>', unsafe_allow_html=True)
        if n2.button(back_label, key=f"back-{key}", type="tertiary"):
            state.pop(f"scope-{key}", None); state[f"rerun-{key}"] = True; st.rerun()
    cited_now = cited(answer, len(docs)) if not docs.empty else []
    # Number the sources in the order the answer cites them; uncited records follow.
    first_seen = [int(n) for n in re.findall(r"\[(\d+)\]", answer or "") if int(n) in cited_now]
    order = list(dict.fromkeys(first_seen)) + [n for n in range(1, len(docs) + 1) if n not in first_seen]
    disp = {orig: k for k, orig in enumerate(order, 1)}
    with st.container(key=f"ansblock-{key}-{state.get(f'ansn-{key}', 0)}"):
        a1 = a2 = st.container(key=f"anshdr-{key}", horizontal=True, vertical_alignment="center", gap="small")
        _k = len(cited_now) if answer is not None else 0
        _count = "no sources" if _k == 0 else ("1 source" if _k == 1 else f"{_k} sources")
        a1.markdown(f'<div class="rim-card-h" role="heading" aria-level="2">Answer from your records <span class="rim-count">· {_count}</span></div>',
                    unsafe_allow_html=True, width="stretch")
        save_item = {"kind": "question", "question": q, "panel": key, "scope": state.get(f"scope-{key}"),
                     "node": list(state.node) if key.startswith("folder-") else None,
                     "record": key[4:] if key.startswith("rec-") else None}
        save_popover(a2, q, save_item, f"ans-{key}", width="content")
        if answer is None:
            st.markdown(f'<div class="rim-warn">{NO_OLLAMA}</div>', unsafe_allow_html=True)
        else:
            if state.pop(f"announce-{key}", False):
                k = len(cited_now)
                said = ("Answer ready. No sources listed." if k == 0 else
                        f"Answer ready, with {k} source{'s' if k != 1 else ''} listed below.")
                st.markdown(f'<div class="rim-sr" role="status">{said}</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="rim-answer">{answer_html(answer, disp)}</div>', unsafe_allow_html=True)
            st.caption(f"{LLM} on this computer · only from the records below")
        # Follow-ups only when the answer cites sources; without Ollama, related records of the closest ones
        if not docs.empty and (cited_now or answer is None):
            base = docs.iloc[[n - 1 for n in cited_now]] if cited_now else docs
            actions = []
            if cited_now:
                counts = base["department_name"].value_counts()
                top = counts[counts == counts.iloc[0]].index.tolist()[:2]
                scope_ids = set(ids)
                for dept in top:
                    dept_ids = [i for i in archive.loc[archive["department_name"] == dept, "id"] if i in scope_ids]
                    if len(dept_ids) < len(scope_ids):             # not offered when the scope is already this department
                        actions.append(("dept", dept, dept_ids))
            actions.append(("related", None, None))
            if cited_now and not (scope and scope["kind"] == "sources"):
                actions.append(("sources", None, None))
            fu = st.container(key=f"followups-{key}", horizontal=True, gap="small", vertical_alignment="center")
            fu.markdown('<span class="rim-eyebrow" role="heading" aria-level="3">Follow up</span>', unsafe_allow_html=True, width="content")
            cols = [fu] * len(actions)
            for col, (kind, dept, dept_ids) in zip(cols, actions):
                if kind == "dept":
                    if col.button(f"Only in {dept}", key=f"fu-dept-{key}-{dept}", type="tertiary"):
                        state[f"scope-{key}"] = {"kind": "dept", "label": dept, "ids": dept_ids}
                        state[f"rerun-{key}"] = True; st.rerun()
                elif kind == "related":
                    if col.button("Find related records", key=f"fu-rel-{key}", type="tertiary"):
                        state[f"rel-{key}"] = True; st.rerun()
                else:
                    n = len(base)
                    if col.button("Ask about this source only" if n == 1 else f"Ask about these {n} sources only",
                                  key=f"fu-src-{key}", type="tertiary"):
                        state[f"scope-{key}"] = {"kind": "sources", "ids": base["id"].tolist(),
                                                 "label": f"the {n} cited source{'s' if n != 1 else ''}"}
                        state.pop(f"rel-{key}", None); st.rerun()
    if docs.empty:
        return
    cited_n = cited_now
    numbered = docs.assign(n=range(1, len(docs) + 1))
    numbered = numbered.assign(shown_n=numbered["n"].map(disp)).sort_values("shown_n")
    cited_docs = numbered[numbered["n"].isin(cited_n)]
    others = numbered[~numbered["n"].isin(cited_n)]
    shown = cited_docs if len(cited_docs) else numbered

    def source_rows(rows, is_cited=False):
        # one row per record: number, thumbnail, title, date and file ID, filing tag, Open
        for d in rows.itertuples():
            with st.container(key=f"srow-{key}-{d.id}"):
                c0, c1, c2, c3 = st.columns([0.32, 0.5, 5, 1], vertical_alignment="center")
                c0.markdown(f'<span class="rim-srcno">[{d.shown_n}]</span>', unsafe_allow_html=True)
                img = ROOT / d.image if getattr(d, "image", "") else None
                if img is not None and img.exists():
                    c1.image(str(img), width=44)
                filing = d.department_name + (f" › {d.record_type_name}" if d.record_type else "")
                tag = ' <span class="rim-tag">suggested</span>' if d.suggested == "True" else ""
                c2.markdown(f'<div class="rim-src-t">{html.escape(d.title)}</div><div class="rim-src-m">{d.date_shown} · '
                            f'<span class="rim-mono">{d.file_id}</span></div><span class="rim-ftag">{html.escape(filing)}</span>{tag}',
                            unsafe_allow_html=True)
                opened = c3.button("Open", key=f"src-{key}-{d.id}", use_container_width=True)
            if opened:
                if is_cited and answer:
                    state.hl = {"id": d.id, "sentences": citing_sentences(answer, d.n)}
                else:
                    state.pop("hl", None)
                open_record(d.id, docs["id"])

    with st.container(key=f"srcblock-{key}"):
        h1 = h2 = h3 = st.container(key=f"srchdr-{key}", horizontal=True, vertical_alignment="center", gap="small")
        h1.markdown(f'<div class="rim-card-h" role="heading" aria-level="2">Sources cited <span class="rim-count">· {len(cited_docs)}</span></div>' if len(cited_docs)
                    else f'<div class="rim-card-h" role="heading" aria-level="2">Closest records <span class="rim-count">· {len(docs)}</span> '
                         f'<span class="rim-count" style="font-weight:400">(none used in the answer)</span></div>', unsafe_allow_html=True, width="stretch")
        pdf = bundle_pdf(shown)
        if pdf:
            h2.download_button("Download documents (PDF)", pdf, file_name=f"sources_{datetime.now():%Y-%m-%d_%H%M}.pdf",
                               mime="application/pdf", width="content", key=f"pdf-{key}")
        h3.download_button("Download list (CSV)", sources_csv(q, shown.assign(n=shown["shown_n"])), file_name=f"sources_{datetime.now():%Y-%m-%d_%H%M}.csv",
                           mime="text/csv", width="content", key=f"csv-{key}")
        source_rows(shown, is_cited=bool(len(cited_docs)))
        if len(cited_docs) and len(others):
            with st.expander(f"Other records searched · {len(others)} (not cited)"):
                source_rows(others)

    if state.get(f"rel-{key}"):
        rel = related_to(shown["id"].tolist(), set(shown["id"]))
        with st.container(key=f"relblock-{key}"):
            r1, r2 = st.columns([5, 1], vertical_alignment="center")
            r1.markdown(f'<div class="rim-card-h" role="heading" aria-level="2">{"Related to the sources" if len(cited_docs) else "Related to the closest records"} '
                        f'<span class="rim-count">· {len(rel)}</span></div>', unsafe_allow_html=True)
            if r2.button("Close", key=f"rel-close-{key}", type="tertiary"):
                state.pop(f"rel-{key}", None); st.rerun()
            st.caption("Similar in meaning to the cited records; none of them was used in the answer." if len(cited_docs)
                       else "Similar in meaning to the closest records above; none was used in an answer.")
            for d in rel.itertuples():
                c2, c3 = st.columns([5.35, 1], vertical_alignment="center")
                filing = d.department_name + (f" › {d.record_type_name}" if d.record_type else "")
                tag = ' <span class="rim-tag">suggested</span>' if d.suggested == "True" else ""
                c2.markdown(f'<b>{html.escape(d.title)}</b><br><span class="rim-muted">{d.date_shown} · '
                            f'<span class="rim-mono">{d.file_id}</span> · {html.escape(filing)}{tag}</span>', unsafe_allow_html=True)
                if c3.button("Open", key=f"rel-{key}-{d.id}", use_container_width=True):
                    state.pop("hl", None)
                    open_record(d.id, rel["id"])
