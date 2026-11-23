"""LocalRIM Streamlit app.

Screens: Archive (home, Browse, Reader and the assistant), Intake and the Review queue,
plus an Add files dialog. The archive is outputs/app/archive.csv (built by
app/build_archive.py) plus the records filed in the app (see localrim.intake).
Corrections made in the Reader are logged in outputs/app/record_history.csv and applied
on load; the original filing is never overwritten. Documents never leave the computer.

Usage:
    python app/build_archive.py
    streamlit run app/app.py
"""

from __future__ import annotations

import html
import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as _components
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
from localrim import intake as intake_db
from localrim.assistant import LLM, ask_llm, cited, retrieve as _retrieve, type_words
from localrim.intake import REASONS

APP = ROOT / "outputs" / "app"
HISTORY = APP / "record_history.csv"
TAXONOMY = ROOT / "taxonomy" / "rim_taxonomy_final.yaml"
EMB = ROOT / "results" / "embeddings"
PER_PAGE = 24
DEMO_USER = os.environ.get("LOCALRIM_USER", "Local user")   # name stamped on uploads, approvals and corrections

st.set_page_config(page_title="LocalRIM", page_icon="🗂", layout="wide")

st.markdown("""
<style>
/* Colour tokens. Mulberry (accent): interface only. Topaz (person): pages with a person. Taupe (ref): context. */
@font-face { font-family: "Literata"; font-style: normal; font-weight: 400 700; font-display: swap;
  src: url("/app/static/fonts/literata-latin-var-normal.woff2") format("woff2"); }
@font-face { font-family: "Atkinson Hyperlegible Next"; font-style: normal; font-weight: 400 700; font-display: swap;
  src: url("/app/static/fonts/atkinson-hyperlegible-next-latin-var.woff2") format("woff2"); }
:root { --paper: #F7F2EF; --surface: #FFFCFA; --ink: #2A1F1B; --muted: #6B5A51; --rule: #CDBAB0; --grid: #EBE2DD;
  --tab: #E6DAD4; --person: #D39539; --person-line: #8A5A12; --accent: #C90184; --accent-soft: #F8E6F0; --ref: #8F7B70;
  --hatch-a: rgba(211,149,57,.16); --hatch-b: rgba(211,149,57,.07); --rail: #EFE6E1;
  --text: "Literata", Georgia, serif; --label: "Atkinson Hyperlegible Next", "Segoe UI", sans-serif; }
/* Dark tokens. data-rim-theme is set by the theme script below from the app's real background;
   until it runs, the system setting decides. */
:root[data-rim-theme="dark"] { --paper: #1B1614; --surface: #241D1A; --ink: #F3EBE6; --muted: #BCA99F; --rule: #4A3C35; --grid: #2E2622;
    --tab: #352B26; --person: #D39539; --person-line: #D39539; --accent: #F263B9; --accent-soft: #33202B; --ref: #9C877C;
    --hatch-a: rgba(211,149,57,.24); --hatch-b: rgba(211,149,57,.11); --rail: #211A17; }
@media (prefers-color-scheme: dark) { :root:not([data-rim-theme="light"]) { --paper: #1B1614; --surface: #241D1A; --ink: #F3EBE6; --muted: #BCA99F; --rule: #4A3C35; --grid: #2E2622;
    --tab: #352B26; --person: #D39539; --person-line: #D39539; --accent: #F263B9; --accent-soft: #33202B; --ref: #9C877C;
    --hatch-a: rgba(211,149,57,.24); --hatch-b: rgba(211,149,57,.11); --rail: #211A17; } }
/* a 1 px edge so scans with dark borders don't blur into a dark background */
:root[data-rim-theme="dark"] [data-testid="stImage"] img { outline: 1px solid var(--rule); }
@media (prefers-color-scheme: dark) { :root:not([data-rim-theme="light"]) [data-testid="stImage"] img { outline: 1px solid var(--rule); } }
/* active tab label in the accent text colour (Streamlit's fill colour is too dark on dark paper) */
[data-testid="stTabs"] [role="tab"][aria-selected="true"], [data-testid="stTabs"] [role="tab"][aria-selected="true"] p { color: var(--accent); }
html, body, [class*="css"], .stMarkdown, .stButton button, input, textarea, select { font-family: var(--label); }
.stApp { background: var(--paper); color: var(--ink); }
.block-container { padding-top: 3rem; max-width: 1560px; }
header[data-testid="stHeader"] { height: 0; }
.rim-mark { font-family: var(--label); font-weight: 700; font-size: 22px; color: var(--accent); letter-spacing: -.2px; }
.rim-mark span { color: var(--accent); }
.rim-sub { color: var(--muted); font-size: 14px; }
.rim-eyebrow { font-size: 13px; color: var(--muted); font-weight: 700; }
.rim-h1 { font-family: var(--text); font-size: 30px; font-weight: 600; margin: 0 0 4px; line-height: 1.25; }
.rim-h2 { font-family: var(--text); font-size: 20px; font-weight: 600; margin: 0 0 6px; line-height: 1.3; }
.rim-muted { color: var(--muted); font-size: 14px; }
.rim-mono { font-family: var(--label); font-variant-numeric: tabular-nums lining-nums; letter-spacing: .2px; }
.rim-date { color: var(--ink); font-weight: 700; font-size: 13px; font-variant-numeric: tabular-nums lining-nums; }
.rim-card-title { font-family: var(--text); font-weight: 500; line-height: 1.35; font-size: 14.5px; height: 40px; overflow: hidden; }
.rim-tag { display: inline-block; padding: 1px 8px; border-radius: 4px; font-size: 12.5px; font-weight: 700; background: var(--tab); color: var(--muted); }
.rim-tag-x { display: inline-block; padding: 1px 8px; border-radius: 4px; font-size: 12.5px; font-weight: 700; border: 1px dashed var(--muted); color: var(--muted); }
/* "With a person" marker: plain surface, 3 px topaz top edge and a 12 px hatched swatch before the label */
.rim-warn { display: flex; align-items: flex-start; gap: 8px; background: var(--surface); border: 1px solid var(--rule);
  border-top: 3px solid var(--person-line); border-radius: 0 0 8px 8px; color: var(--ink); padding: 10px 12px; font-size: 14px; line-height: 1.4; }
.rim-warn::before, .st-key-rq-suggest .rim-card-h::before, .st-key-status-review button p::after {
  content: ""; flex: none; display: inline-block; width: 12px; height: 12px; border-radius: 2px;
  background: repeating-linear-gradient(45deg, var(--person) 0 2px, var(--surface) 2px 4px); box-shadow: inset 0 0 0 1px var(--person-line); }
.rim-warn::before { margin-top: 4px; }
.rim-file { font-family: var(--label); font-variant-numeric: tabular-nums; font-size: 12.5px; background: var(--surface); border: 1px solid var(--rule); border-radius: 6px; padding: 8px; word-break: break-all; }
.rim-grid { display: grid; grid-template-columns: 120px 1fr; gap: 6px 12px; font-size: 14px; }
.rim-grid span:nth-child(odd) { color: var(--muted); }
.rim-text { font-family: Consolas, "Courier New", monospace; font-size: 13px; line-height: 1.75; white-space: pre-wrap; max-width: 92ch; background: var(--surface); border: 1px solid var(--rule); border-radius: 6px; padding: 14px 16px; max-height: 760px; overflow-y: auto; }
.rim-text mark { background: none; color: var(--ink); outline: 1.5px solid var(--ink); outline-offset: 1px; border-radius: 2px; }
/* lines that match the answer */
.rim-hl { display: inline-block; width: 100%; background: var(--accent-soft); box-shadow: inset 3px 0 0 var(--accent); padding-left: 8px; margin-left: -8px; scroll-margin-top: 80px; }
.rim-hlbar { font-size: 14px; color: var(--ink); }
.rim-hlbar b { font-weight: 700; }
.rim-hlbar a { display: inline-block; min-width: 24px; padding: 2px 6px; margin: 0 1px; text-align: center; font-weight: 700; text-decoration: underline; }
div[data-testid="stVerticalBlockBorderWrapper"] { background: var(--surface); border-radius: 8px; }
div[data-testid="stMetricValue"] { color: var(--accent); font-family: var(--label); font-weight: 600; font-variant-numeric: tabular-nums lining-nums; }
.stButton button, .stDownloadButton button { border-radius: 6px; min-height: 44px; }
.stButton button:active { transform: scale(.97); }
a { color: var(--accent); }

/* no Deploy button */
[data-testid="stAppDeployButton"] { display: none !important; }
/* placeholders and captions in the muted colour, for contrast */
input::placeholder, textarea::placeholder { color: var(--muted) !important; opacity: 1 !important; }
[data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] p { color: var(--muted) !important; }
/* selected segmented-control option: ink text with a 2 px accent ring */
[data-testid="stButtonGroup"] button[aria-checked="true"] { background: var(--surface) !important; color: var(--ink) !important;
  box-shadow: inset 0 0 0 2px var(--accent) !important; border-color: var(--accent) !important; }
/* empty states */
.rim-empty { border: 1px dashed var(--rule); border-radius: 6px; padding: 14px 16px; color: var(--ink); font-size: 15px; }
/* masthead navigation: text links, the current screen underlined */
.st-key-rim-nav button[kind="tertiary"] { min-height: 44px; padding: 0 4px; border-radius: 0; color: var(--muted); font-weight: 600; white-space: nowrap; }
.st-key-rim-nav button[kind="tertiary"]:hover { color: var(--ink); }
.st-key-rim-nav [class*="st-key-nav-on-"] button { color: var(--ink); box-shadow: inset 0 -3px 0 var(--accent); }
.st-key-rim-nav button[kind="primary"] { white-space: nowrap; padding: 0 16px; }
.rim-rule { height: 1px; background: var(--rule); margin: 6px 0 16px; }
/* row lists (home departments, Browse tree, Recently filed): left-aligned, borderless, tinted on hover */
[class*="st-key-rim-list"] button, .st-key-rim-tree button { justify-content: flex-start; text-align: left; border: 0; background: none;
  min-height: 36px; padding: 4px 8px; border-radius: 4px; color: var(--ink); font-weight: 500; }
[class*="st-key-rim-list"] button > div, .st-key-rim-tree button > div { justify-content: flex-start; }
[class*="st-key-rim-list"] button p, .st-key-rim-tree button p { text-align: left; }
[class*="st-key-rim-list"] button:hover, .st-key-rim-tree button:hover { background: var(--tab); color: var(--ink); }
.st-key-rim-tree button[kind="primary"] { background: var(--accent-soft); color: var(--accent); font-weight: 700; }
.st-key-rim-tree [class*="st-key-t-"] button { padding-left: 22px; font-weight: 400; }
.st-key-rim-tree [class*="st-key-s-"] button { font-weight: 600; }
/* Archive home status line */
.rim-status { font-size: 15px; color: var(--muted); }
.rim-status b { color: var(--accent); font-weight: 600; font-variant-numeric: tabular-nums lining-nums; }
.st-key-rim-status button { min-height: 32px; padding: 0 2px; font-size: 15px; }
/* listed answers as a real list */
.rim-answer { font-family: var(--text); font-size: 17px; line-height: 1.65; }
.rim-answer ul { margin: 6px 0 0; padding-left: 1.2em; }
.rim-answer li { margin: 2px 0; }

/* masthead in one row; below 1100 px the client name moves under the wordmark */
.rim-sub-block { display: block; margin-top: 2px; } .rim-slash { display: none; }
.st-key-nav-add { display: flex; justify-content: flex-end; } .st-key-nav-add button { white-space: nowrap; padding: 0 14px; }
@media (min-width: 1200px) { .rim-user { white-space: nowrap; } }
@media (max-width: 1100px) { .rim-user { font-size: 13px; } .st-key-rim-nav { gap: 0.6rem !important; } }
.rim-answer { max-width: 70ch; }
/* Reader top bar */
.st-key-rdr-back button { color: var(--accent); padding: 0 2px; font-weight: 600; }
.st-key-rdr-nav { justify-content: center; }
.st-key-rdr-view { justify-content: flex-end; }
/* Browse: department groups without a box */
.st-key-rim-tree [data-testid="stExpander"] details { border: 0; background: none; }
.st-key-rim-tree [data-testid="stExpander"] summary { padding-left: 8px; font-weight: 600; }
.st-key-rim-tree [data-testid="stExpanderDetails"] { padding: 0 0 4px 6px; }
/* Archive home: status lines in the left column */
.st-key-rim-status-col .rim-status { display: block; margin: 0; line-height: 1.6; }
.st-key-rim-status-col button { min-height: 28px; padding: 0; justify-content: flex-start; color: var(--ink); }
.st-key-rim-status-col button p { text-align: left; font-size: 15px; }
.st-key-rim-status-col button b, .st-key-rim-status-col button strong { color: var(--accent); }
/* Review queue panel */
.rq-warn { padding: 6px 10px; }
.st-key-rq-suggest { padding: 10px 12px; }
.rq-reason { font-size: 14px; line-height: 1.5; color: var(--ink); margin: 0; }
.st-key-rq-td { flex-wrap: wrap; }
.st-key-rq-td [class*="st-key-rq-title-"] { flex: 1 1 220px; min-width: 220px; }
.rq-type { font-weight: 700; line-height: 1.35; margin: 0; }

/* Accessibility */
/* hide Streamlit's heading anchor icons */
[data-testid="stHeaderActionElements"], .stMarkdown a.anchor-link, [data-testid="stHeadingWithActionElements"] a[href^="#"] { display: none !important; }
/* keyboard focus: solid 2 px accent ring, 2 px off the control */
button:focus-visible, a:focus-visible, [role="tab"]:focus-visible, [role="radio"]:focus-visible, summary:focus-visible,
.stTextInput div:has(> input):focus-within, .stSelectbox [role="group"]:focus-within,
input[type="checkbox"]:focus-visible + div {
  outline: 2px solid var(--accent) !important; outline-offset: 2px !important; box-shadow: none !important; }
/* visible to screen readers only */
.rim-sr { position: absolute !important; width: 1px; height: 1px; padding: 0; margin: -1px; overflow: hidden;
  clip: rect(0 0 0 0); clip-path: inset(50%); white-space: nowrap; border: 0; }
/* field edges */
.stTextInput div:has(> input), .stSelectbox [role="group"] { border: 1px solid var(--ref) !important; }

/* one filled button per screen: Add files and the Reader's Ask are outlined */
.st-key-nav-add button, [class*="st-key-FormSubmitter-ask-rec-"] button {
  background: var(--surface); color: var(--accent); border: 1px solid var(--accent); font-weight: 600; }
.st-key-nav-add button:hover, [class*="st-key-FormSubmitter-ask-rec-"] button:hover {
  background: var(--accent-soft); color: var(--accent); border-color: var(--accent); }
/* control heights: 44 px standalone, 40 px in lists */
[data-testid="stButtonGroup"] button, [data-testid="stPopover"] button, [data-testid="stFormSubmitButton"] button { min-height: 44px; }
.stTextInput div:has(> input), .stSelectbox [role="group"] { min-height: 44px; }
[class*="st-key-rim-list"] button, .st-key-rim-tree button, [class*="st-key-followups-"] button { min-height: 40px; }
.st-key-rim-status-col button { min-height: 40px; }
.st-key-rim-status-col .rim-status { display: block; min-height: 40px; padding: 9px 0; line-height: 1.4; }
/* starter questions as text links */
[class*="st-key-starters-"] button { color: var(--accent); text-decoration: underline; text-underline-offset: 3px; font-weight: 600;
  min-height: 40px; padding: 0; justify-content: flex-start; text-align: left; }
[class*="st-key-starters-"] button p { text-align: left; }
[class*="st-key-starters-"] button:hover { color: var(--ink); }
/* row text lines up with the text above it; the hover tint reaches into the margin */
.st-key-rim-list { margin-left: -8px; width: calc(100% + 8px); }
.rim-idline { white-space: nowrap; }
[class*="st-key-fbar-"] { flex-wrap: wrap; }
/* similar filed pages: a numbered list of links */
[class*="st-key-simlist-"] button { color: var(--accent); min-height: 40px; padding: 0; justify-content: flex-start; text-align: left; font-size: 14px; }
[class*="st-key-simlist-"] button p { text-align: left; font-size: 14px; text-decoration: underline; text-underline-offset: 3px; }
[class*="st-key-simlist-"] button:hover { color: var(--ink); }
.rim-sim-x { font-size: 14px; color: var(--muted); min-height: 40px; display: flex; align-items: center; }
[class*="st-key-fbar-"] [class*="st-key-in-"] { flex: 1 1 240px; min-width: 240px; }

/* Zones: each kind of content sits in its own bordered surface. Interface text is sans serif;
   answers use the serif. */
.rim-h1 { font-family: var(--label); font-size: 24px; font-weight: 700; line-height: 1.25; }
.rim-h2 { font-family: var(--label); font-size: 18px; font-weight: 700; line-height: 1.3; }
.rim-card-h { font: 700 15px/1.3 var(--label); color: var(--ink); margin: 0; }
.rim-card-h .rim-count { font-weight: 600; color: var(--muted); }
/* a zone: surface, 1 px rule, 8 px corners */
[class*="st-key-zone-"], [class*="st-key-ansblock-"], [class*="st-key-srcblock-"], [class*="st-key-relblock-"],
.st-key-rq-suggest, .st-key-rim-preview {
  background: var(--surface); border: 1px solid var(--rule) !important; border-radius: 8px; padding: 14px 16px; margin-top: 0; }
/* the answer card: 3 px accent edge on top */
[class*="st-key-ansblock-"] { border-top: 3px solid var(--accent) !important; border-radius: 0 0 8px 8px; }
[class*="st-key-srcblock-"], [class*="st-key-relblock-"] { margin-top: 0; padding-top: 12px; }
/* page viewer background */
[class*="st-key-zone-viewer"] { background: var(--grid); }
/* app bar: a surface band across the window */
.st-key-rim-appbar { background: var(--surface); box-shadow: 0 0 0 100vmax var(--surface); clip-path: inset(-1rem -100vmax 0 -100vmax);
  border-bottom: 1px solid var(--rule); padding: 4px 0 6px; margin: -2rem 0 6px; }
/* left rail */
.st-key-rim-rail { background: var(--rail); border: 1px solid var(--rule); border-radius: 8px; padding: 14px 12px; }
.rim-stat { min-height: 48px;
  background: var(--surface); border: 1px solid var(--rule); border-radius: 8px; padding: 8px 12px; }
.rim-stat b { font: 600 24px/1.1 var(--label); color: var(--accent); font-variant-numeric: tabular-nums lining-nums; }
.rim-stat span { font-size: 13.5px; line-height: 1.3; color: var(--ink); }
.st-key-status-review button { display: block; width: 100%; min-height: 48px; padding: 8px 12px; text-align: left; border-radius: 0 0 8px 8px;
  border: 1px solid var(--rule); border-top: 3px solid var(--person-line); background: var(--surface); color: var(--ink); }
/* stat rows: number | swatch | label, so every label starts on the same line */
.st-key-status-review button p, .rim-stat { display: grid; grid-template-columns: 52px 18px minmax(0, 1fr); align-items: center; column-gap: 6px; row-gap: 2px; text-align: left; }
.st-key-status-review button p { font-size: 13.5px; }
.st-key-status-review button p strong { grid-column: 1; grid-row: 1; font: 600 24px/1.1 var(--label); color: var(--accent); }
.st-key-status-review button p::after { grid-column: 2; grid-row: 1; }
.rim-stat b { grid-column: 1; } .rim-stat span { grid-column: 3; }
.st-key-status-review button:hover { border-color: var(--accent); color: var(--ink); }
.rim-rail-h { font: 700 12.5px/1.3 var(--label); color: var(--muted); margin: 14px 0 2px; }
/* department rows: name, count badge, share bar */
[class*="st-key-dept-"] { border-radius: 6px; padding: 2px 6px 6px; gap: 0 !important; }
[class*="st-key-dept-"]:hover { background: var(--surface); }
[class*="st-key-deptrow-"] { align-items: center; flex-wrap: nowrap !important; }
[class*="st-key-deptrow-"] > div:first-child { flex: 1 1 auto; min-width: 0; }
[class*="st-key-deptrow-"] button > div { justify-content: flex-start; }
[class*="st-key-deptrow-"] button p, [class*="st-key-deptrow-"] button div { white-space: normal !important; overflow: visible !important; text-overflow: clip !important; line-height: 1.3; }
[class*="st-key-deptrow-"] button { justify-content: flex-start; text-align: left; border: 0; background: none; min-height: 36px; padding: 2px 2px;
  color: var(--ink); font-weight: 500; }
[class*="st-key-deptrow-"] button p { text-align: left; }
.rim-badge { display: inline-block; font: 600 12.5px/1.3 var(--label); color: var(--ink); background: var(--tab); border-radius: 10px;
  padding: 2px 9px; font-variant-numeric: tabular-nums; white-space: nowrap; }
.rim-share { height: 3px; border-radius: 2px; background: var(--grid); overflow: hidden; margin: 0 2px; }
.rim-share i { display: block; height: 100%; background: var(--ref); }
/* follow-up buttons in the answer card's footer */
[class*="st-key-followups-"] { border-top: 1px solid var(--grid); padding-top: 10px; }
[class*="st-key-followups-"] button { color: var(--ink) !important; text-decoration: none !important; font-weight: 600; min-height: 36px !important;
  padding: 0 12px !important; border: 1px solid var(--rule) !important; background: var(--surface) !important; border-radius: 6px; }
[class*="st-key-followups-"] button:hover { border-color: var(--accent) !important; color: var(--accent) !important; }
/* source rows: thumbnail, title, meta, filing tag, Open */
[class*="st-key-srow-"] { border-top: 1px solid var(--grid); padding-top: 8px; }
.rim-src-t { font: 600 15px/1.35 var(--label); color: var(--ink); }
.rim-src-m { font-size: 13px; color: var(--muted); margin-top: 1px; }
.rim-ftag { display: inline-block; margin-top: 4px; font: 600 12.5px/1.35 var(--label); padding: 2px 8px; border-radius: 4px; background: var(--tab); color: var(--ink); }
.rim-srcno { font: 700 13px var(--label); color: var(--accent); }
[class*="st-key-srow-"] [data-testid="stImage"] img { border: 1px solid var(--rule); border-radius: 3px; }
/* Recently filed rows */
.st-key-rim-list-recent [data-testid="stHorizontalBlock"] { border-top: 1px solid var(--grid); padding-top: 4px; }
/* review card: topaz top edge and swatch */
.st-key-rq-suggest { border-top: 3px solid var(--person-line) !important; border-radius: 0 0 8px 8px; }
.st-key-rq-suggest .rim-card-h { display: flex; align-items: center; gap: 8px; }
.rq-why { font-size: 13.5px; color: var(--ink); margin: 2px 0 0; }
/* narrow windows: the stat number sits above its label and words wrap only at spaces */
.st-key-rim-rail, .st-key-rim-rail * { word-break: normal !important; overflow-wrap: normal !important; hyphens: none; }
@media (max-width: 1100px) {
  .rim-stat, .st-key-status-review button p { grid-template-columns: 12px minmax(0, 1fr); column-gap: 6px; }
  .rim-stat b, .st-key-status-review button p strong { grid-column: 1 / -1; }
  .rim-stat span { grid-column: 2; }   /* aligned with the queue row's label */
  .st-key-status-review button p::after { grid-column: 1; grid-row: 2; }
}
/* header rows wrap instead of cutting their buttons */
[class*="st-key-anshdr-"], [class*="st-key-srchdr-"] { flex-wrap: wrap; row-gap: 8px; }
/* Intake: the batch picker drops under the title when there's no room */
.st-key-zone-inbar { flex-wrap: wrap; row-gap: 10px; }
.st-key-zone-inbar > [data-testid="stLayoutWrapper"]:has(> .st-key-inbar-head) { flex: 1 1 380px; min-width: 300px; }   /* set on the wrapper; inside it, flex would set the height */
/* quick view bar */
.st-key-qv-bar { border-bottom: 1px solid var(--grid); padding-bottom: 8px; }
/* Browse folder tree styled as a rail */
.st-key-rim-tree { background: var(--rail); border: 1px solid var(--rule); border-radius: 8px; padding: 14px 12px; }
.st-key-rim-tree [data-testid="stExpander"] details { background: none; }
/* headings at the top of a zone */
[class*="st-key-zone-"] .rim-h1, [class*="st-key-zone-"] .rim-h2 { margin-top: 0 !important; }
/* Add files dialog headings in sans serif */
[data-testid="stDialog"] h1, [data-testid="stDialog"] h2, [data-testid="stDialog"] h3 { font-family: var(--label); font-weight: 700; }

/* Motion: only when something opens, arrives or changes screen, once per event. Animations use
   fill "backwards" so no transform is left behind on fixed or sticky children. */
:root { --ease-out: cubic-bezier(.23, 1, .32, 1); }
/* rim-in stays invisible for the first 60 ms: on a screen change Streamlit briefly shows the old
   screen's content in the new container (up to about 75 ms) before replacing it */
@keyframes rim-in { 0%, 25% { opacity: 0; transform: translateY(4px); } 100% { opacity: 1; transform: none; } }
@keyframes rim-rise { from { opacity: 0; transform: translateY(6px); } }
@keyframes rim-fade { from { opacity: 0; } }
@keyframes rim-open { from { opacity: 0; transform: scale(.97); } }
/* a new screen fades in (the container is keyed by screen, so ordinary reruns don't replay it) */
[class*="st-key-screen-"] { animation: rim-in 240ms var(--ease-out) backwards; }
/* a new answer rises in once (keyed by answer) */
[class*="st-key-ansblock-"] { animation: rim-rise 240ms var(--ease-out) backwards; }
/* Browse preview fades in when it first opens */
.st-key-rim-preview { animation: rim-fade 160ms var(--ease-out) backwards; }
/* Add files dialog opens from .97; the backdrop fades */
[data-testid="stDialog"] [role="dialog"] { animation: rim-open 200ms var(--ease-out) backwards; transform-origin: center; }
[data-testid="stDialog"] > div { animation: rim-fade 200ms var(--ease-out) backwards; }
/* press feedback */
.stButton button { transition: transform 120ms var(--ease-out); }
/* Streamlit's stale-content fade, after the same 0.5 s delay */
[data-stale="true"] { transition: opacity 200ms var(--ease-out) 500ms !important; }
/* the helper iframes take no space */
[data-testid="stElementContainer"]:has(iframe[srcdoc*="rim-helper"]) { display: none; }
/* reduced motion: no animations or transitions */
@media (prefers-reduced-motion: reduce) {
  *, *::before, *::after { animation-duration: .01ms !important; animation-delay: 0ms !important; animation-iteration-count: 1 !important;
    transition-duration: .01ms !important; transition-delay: 0ms !important; }
  .stButton button:active { transform: none; }
  [data-stale="true"] { transition: none !important; }   /* overrides the stale-content transition above */
}
</style>
""", unsafe_allow_html=True)


# Mark the page light or dark from the app's background, so the CSS colour tokens follow the theme.
# The script never changes, so Streamlit keeps the iframe and it re-checks every 400 ms.
_components.html("""<script>/* rim-helper theme */
const P = window.parent.document;
function mark() {
  const app = P.querySelector('.stApp'); if (!app) return;
  const m = getComputedStyle(app).backgroundColor.match(/\\d+/g); if (!m) return;
  const lum = (0.2126 * m[0] + 0.7152 * m[1] + 0.0722 * m[2]) / 255;
  const t = lum < 0.5 ? 'dark' : 'light';
  if (P.documentElement.dataset.rimTheme !== t) P.documentElement.dataset.rimTheme = t;
}
mark(); setInterval(mark, 400);
</script>""", height=0)


def show_image(path, **kw):
    """Show an image at the column width (older Streamlit versions lack use_container_width)."""
    try:
        st.image(str(path), use_container_width=True, **kw)
    except TypeError:
        st.image(str(path), **kw)


# Data

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


archive_built, queue, tax = load_static()
_filed = intake_db.read(intake_db.FILED)
archive_raw = pd.concat([archive_built, _filed], ignore_index=True).fillna("") if len(_filed) else archive_built
archive_raw = archive_raw[~archive_raw["id"].isin(intake_db.HIDDEN)].reset_index(drop=True)
queue = queue[~queue["id"].isin(intake_db.HIDDEN)].reset_index(drop=True)
dept_name = {f["id"]: f["name"] for f in tax["functions"]}
type_info = {}
for f in tax["functions"]:
    for s in f["series"]:
        for t in s["types"]:
            type_info[t["id"]] = (f["id"], f["name"], s["id"], s["name"], t["name"])


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


archive = apply_history(archive_raw)
state = st.session_state
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


for k, v in {"tab": "Archive", "view": "home", "node": ("all", "", "All records"),
             "record": None, "list": [], "mode": "Page", "shown": PER_PAGE, "pdf_page": 1}.items():
    state.setdefault(k, v)


def go(view, **kw):
    """Switch to an Archive view and set any extra state."""
    state.view = view
    for k, v in kw.items():
        state[k] = v


# Current user. A real deployment would take this from its sign-in.
state.user = DEMO_USER
if "mode_loaded" not in state:
    _saved_view = _read_prefs().get(state.user)
    if _saved_view in READER_VIEWS:
        state.mode = _saved_view
    state.mode_loaded = True


# Header
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


# Masthead: wordmark, navigation, Add files and the current user.
_appbar = st.container(key="rim-appbar")
h1, h2, h5, h6 = _appbar.columns([1.5, 3.7, 1.5, 1.9], vertical_alignment="center")
h1.markdown('<span class="rim-mark">LocalRIM</span><span class="rim-sub rim-sub-block"><span class="rim-slash"> / </span>Philip Morris USA records</span>',
            unsafe_allow_html=True)
review_queue = intake_db.queue_view(queue)
_intake_all = intake_db.read(intake_db.INTAKE)
waiting_n = int((_intake_all["status"] == "waiting").sum()) if len(_intake_all) else 0
with h2.container(horizontal=True, key="rim-nav", gap="medium", vertical_alignment="center"):
    for name, label in [("Archive", "Archive"), ("Intake", f"Intake · {waiting_n}" if waiting_n else "Intake"),
                        ("Review", f"Review queue · {len(review_queue)}")]:
        if st.button(label, type="tertiary", key=f"nav-on-{name}" if state.tab == name else f"nav-{name}"):
            state.tab = name
            if name == "Archive":
                state.view = "home"
            st.rerun()
with h5:
    if st.button("⬆ Add files", key="nav-add"):
        state.just_added = None
        add_files()
h6.markdown(f'<div class="rim-muted rim-user" style="text-align:right">Reviewing as <b>{html.escape(state.user)}</b> · on this computer</div>',
            unsafe_allow_html=True)


# Shared pieces
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


# Archive assistant: search by meaning and words, answer with sources
NO_OLLAMA = "Assistant unavailable without Ollama. Showing only the most relevant documents for now."
TYPE_WORDS = type_words({t: info[4] for t, info in type_info.items()})


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


# Archive home
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
            st.markdown(f"**{r['added']}** · {r['how_filed']}")
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




# Intake: batch inventory and approval
TYPE_OPTIONS = sorted(type_info, key=lambda t: (type_info[t][1], type_info[t][3], type_info[t][4]))
TYPE_LABEL = {t: f"{type_info[t][1]} › {type_info[t][3]} › {type_info[t][4]}" for t in TYPE_OPTIONS}
LABEL_TYPE = {v: k for k, v in TYPE_LABEL.items()}


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


# Review queue: one page at a time with the review helper's suggestion
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


CODE_NAME = {**{f["id"]: f["name"] for f in tax["functions"]},
             **{s_["id"]: s_["name"] for f in tax["functions"] for s_ in f["series"]},
             **{t: type_info[t][4] for t in type_info}}
_CODES = "|".join(sorted(map(re.escape, CODE_NAME), key=len, reverse=True))
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


# Scroll to the top on every screen change (Streamlit keeps the old scroll position). The
# signature in the script changes only with the screen, so the iframe re-runs once per change.
_screen = state.tab if state.tab in ("Intake", "Review") else ("reader" if state.view == "reader" and state.record else state.view)
_sig = f"{_screen}|{state.record if _screen == 'reader' else ''}|{state.get('rq', '') if _screen == 'Review' else ''}|{state.get('intake_batch', '') if _screen == 'Intake' else ''}"
_components.html(f"""<script>/* rim-helper top {html.escape(_sig)} */
const m = window.parent.document.querySelector('[data-testid="stMain"]'); if (m) m.scrollTo(0, 0);</script>""", height=0)

# keyed by screen, so a new screen fades in once
with st.container(key=f"screen-{_screen.lower()}"):
    if state.tab == "Intake":
        intake_screen()
    elif state.tab == "Review":
        review_screen()
    elif state.view == "browse":
        browse()
    elif state.view == "reader" and state.record:
        reader()
    else:
        home()
