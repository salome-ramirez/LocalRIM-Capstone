"""Blind spot-check app for 20 auto-filed scale-run pages.

Shows each page drawn by scale_run.py (results/scale_run/spotcheck_sample.csv;
final-test pages left out) without the model's answer; the reviewer picks the department
as when labeling. scale_run.py then reports the agreement (descriptive only).

Saves to labels/spotcheck_<annotator>.csv (the newest save per page counts).

Usage:
    python -m streamlit run pipeline/05_validate/spotcheck.py
"""

from __future__ import annotations

import datetime as dt

import pandas as pd
import streamlit as st
import yaml

from pipeline.common import ROOT, SPOT_LABELS as OUT

SAMPLE = ROOT / "results" / "scale_run" / "spotcheck_sample.csv"
IMAGES = ROOT / "data" / "pm_sample" / "images"
TAXONOMY = ROOT / "taxonomy" / "rim_taxonomy_final.yaml"
EXCLUDE = "Can't tell / exclude"

st.set_page_config(page_title="LocalRIM spot-check", layout="wide")
st.title("Spot-check: auto-filed pages (blind)")

if not SAMPLE.exists():
    st.error("Run python -m pipeline.05_validate.scale_run first: it draws the sample.")
    st.stop()

sample = pd.read_csv(SAMPLE, dtype=str)["id"].tolist()
saved = pd.read_csv(OUT, dtype=str) if OUT.exists() else pd.DataFrame(columns=["doc_id"])
done = set(saved["doc_id"])
todo = [i for i in sample if i not in done]
st.sidebar.metric("Checked", f"{len(sample) - len(todo)} of {len(sample)}")
if not todo:
    st.success("All pages checked. Run python -m pipeline.05_validate.scale_run again to see the agreement.")
    st.stop()

tax = yaml.safe_load(TAXONOMY.read_text(encoding="utf-8"))
departments = {f["id"]: f["name"] for f in tax["functions"]}
doc_id = todo[0]

left, right = st.columns([3, 2])
with left:
    st.image(str(IMAGES / f"{doc_id}.png"), width="stretch")
with right:
    choice = st.radio("Which department keeps the official copy?",
                      list(departments) + [EXCLUDE],
                      format_func=lambda k: k if k == EXCLUDE else f"{departments[k]} ({k})",
                      index=None, key=f"dept_{doc_id}")
    certainty = st.radio("Certainty", ["sure", "unsure"], horizontal=True, key=f"cert_{doc_id}")
    if st.button("Save and next", type="primary", disabled=choice is None):
        row = {"doc_id": doc_id,
               "function_id": "" if choice == EXCLUDE else choice,
               "outcome": "exclude" if choice == EXCLUDE else "labeled",
               "certainty": certainty,
               "saved_at": dt.datetime.now().isoformat(timespec="seconds")}
        pd.DataFrame([row]).to_csv(OUT, mode="a", header=not OUT.exists(), index=False)
        st.rerun()
