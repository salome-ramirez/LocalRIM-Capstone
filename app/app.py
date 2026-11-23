"""LocalRIM Streamlit app: the entry point.

Screens: Archive (home, Browse, Reader and the assistant), Intake and the Review queue,
plus an Add files dialog, one module each in app/screens/. This script sets up the page,
loads the data on every run and hands it to the screens, draws the masthead and shows the
current screen. The archive is outputs/app/archive.csv (built by app/build_archive.py) plus
the records filed in the app (see localrim.intake). Corrections made in the Reader are
logged in outputs/app/record_history.csv and applied on load; the original filing is never
overwritten. Documents never leave the computer.

Usage:
    python -m app.build_archive
    python -m streamlit run app/app.py
"""

from __future__ import annotations

import html
import os
import re

import pandas as pd
import streamlit as st
import streamlit.components.v1 as _components

import screens
from localrim import intake as intake_db
from localrim.assistant import type_words
from screens.intake import add_files, intake_screen
from screens.records import READER_VIEWS, _read_prefs, browse, home, reader
from screens.review import review_screen
from screens.shared import PER_PAGE, apply_history, load_static

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


# Data, loaded on every run and handed to the screens
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
archive = apply_history(archive_raw)
review_queue = intake_db.queue_view(queue)
_intake_all = intake_db.read(intake_db.INTAKE)
waiting_n = int((_intake_all["status"] == "waiting").sum()) if len(_intake_all) else 0
TYPE_WORDS = type_words({t: info[4] for t, info in type_info.items()})
TYPE_OPTIONS = sorted(type_info, key=lambda t: (type_info[t][1], type_info[t][3], type_info[t][4]))
TYPE_LABEL = {t: f"{type_info[t][1]} › {type_info[t][3]} › {type_info[t][4]}" for t in TYPE_OPTIONS}
LABEL_TYPE = {v: k for k, v in TYPE_LABEL.items()}
CODE_NAME = {**{f["id"]: f["name"] for f in tax["functions"]},
             **{s_["id"]: s_["name"] for f in tax["functions"] for s_ in f["series"]},
             **{t: type_info[t][4] for t in type_info}}
_CODES = "|".join(sorted(map(re.escape, CODE_NAME), key=len, reverse=True))
screens.use(archive=archive, archive_raw=archive_raw, queue=queue, tax=tax, dept_name=dept_name, type_info=type_info,
            review_queue=review_queue, waiting_n=waiting_n, TYPE_WORDS=TYPE_WORDS, TYPE_OPTIONS=TYPE_OPTIONS,
            TYPE_LABEL=TYPE_LABEL, LABEL_TYPE=LABEL_TYPE, CODE_NAME=CODE_NAME, _CODES=_CODES)

state = st.session_state
for k, v in {"tab": "Archive", "view": "home", "node": ("all", "", "All records"),
             "record": None, "list": [], "mode": "Page", "shown": PER_PAGE, "pdf_page": 1}.items():
    state.setdefault(k, v)

# Current user. A real deployment would take this from its sign-in.
state.user = DEMO_USER
if "mode_loaded" not in state:
    _saved_view = _read_prefs().get(state.user)
    if _saved_view in READER_VIEWS:
        state.mode = _saved_view
    state.mode_loaded = True


# Masthead: wordmark, navigation, Add files and the current user.
_appbar = st.container(key="rim-appbar")
h1, h2, h5, h6 = _appbar.columns([1.5, 3.7, 1.5, 1.9], vertical_alignment="center")
h1.markdown('<span class="rim-mark">LocalRIM</span><span class="rim-sub rim-sub-block"><span class="rim-slash"> / </span>Philip Morris USA records</span>',
            unsafe_allow_html=True)
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
