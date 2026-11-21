"""Build docs/data/site_data.json, the numbers the project site shows.

Reads the project's labels, taxonomy and result tables and writes counts, scores and
rates only: no label notes, page text, model reasons, names or file paths reach the site.
Missing inputs are skipped, and that part of the site shows "not run".

Reads:
    labels/label_pm_labels_*.csv, skipped_pm_*.csv     training labels and skips
    labels/skim_labels.csv, skim_v2_labels.csv         the two pilot labeling rounds
    labels/final_test_labels_*.csv, spotcheck_*.csv    final-test labels, scale-run spot-check
    data/pm_sample/manifest.csv, page_profile.csv      the sample and its scan-quality groups
    data/pm_sample/ocr_paddle/ocr_results.csv          orientation turns suggested vs kept
    data/pm_sample/ocr_text/sources.csv, ocr_text_clean/   the classifier's text options
    taxonomy/rim_taxonomy_final.yaml
    results/rvl_pilot/                                 the first OCR pilot (result tables only)
    results/funsd_ocr_eval.csv, ocr_eda/, ocr_compare/, page_groups_k_check.csv
    results/classifier/, embeddings/, rag/, final_test/, scale_run/, record_types/
    results/archive_check/                             the app's assistant check
    outputs/app/archive.csv, filed.csv                 the app's archive (app/build_archive.py)

Also writes the batch records (results/batch_metrics.csv) and stops without writing if
they and the site disagree.

Usage:
    python pipeline/06_site_data/build_site_data.py
    python pipeline/06_site_data/build_site_data.py --snapshot NAME [--as-of ISO-TIME]
"""

import json
from datetime import datetime
from pathlib import Path

import sys

import numpy as np
import pandas as pd
import yaml

# The output goes to docs/, where the site reads it.
ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "docs" / "data" / "site_data.json"

sys.path[:0] = [str(ROOT / "app"), str(Path(__file__).resolve().parent)]
# The site and the batch records use the same counting (localrim_metrics.py),
# so their numbers always match.
import batch_metrics            # noqa: E402
import localrim_metrics as lm   # noqa: E402

# Snapshots: --snapshot NAME [--as-of ISO-TIME] writes results/snapshots/site_data_NAME.json
# (and batch_metrics_NAME.csv) instead of the live file. --as-of rebuilds an earlier
# round from the labels saved by then.
args = sys.argv[1:]
SNAPSHOT = args[args.index("--snapshot") + 1] if "--snapshot" in args else None
if "--as-of" in args:
    lm.AS_OF = args[args.index("--as-of") + 1]
SNAP_DIR = ROOT / "results" / "snapshots"
if SNAPSHOT:
    OUT = SNAP_DIR / f"site_data_{SNAPSHOT}.json"

TARGET = 250                 # raised from 200 at the 50-label checkpoint
LABEL_POOL = 400             # manifest positions 1-400 are the labeling pool; 401-403 replace failed
                             # downloads; 404-1,403 are the scale batch (final test + scale run)
LOW_COUNT_RULE = 15          # raise the target to 250 if a department projects under this
DEADLINE = "2026-11-23"

# Archive counts from the UCSF API (Nov 6, 2026); fixed facts, not recomputed.
PM_TOTAL = 5_250_814
PM_PUBLIC = 4_740_682
PM_PUBLIC_SCANNED = 4_466_295

# Department order and display names on the site
DEPT_NAMES = {
    "RND": "Research and development", "QSH": "Quality, safety and health",
    "MKT": "Marketing and sales", "AFF": "Corporate affairs", "LEG": "Legal",
    "FIN": "Finance", "HR": "Human resources", "ADM": "Administration and records",
}
SHORT = {"RND": "R&D", "QSH": "Quality", "MKT": "Marketing", "AFF": "Corp. affairs",
         "LEG": "Legal", "FIN": "Finance", "HR": "HR", "ADM": "Admin"}

# Exclusion reasons, as they start the label note
EXCLUDE_REASONS = {
    "folder cover": "Folder, no contents shown",
    "blank or unreadable": "Blank or unreadable",
    "duplicate": "Duplicate",
    "non-tobacco": "Non-tobacco business",
    "not a business record": "Not a business record",
    "out of scope (hr)": "Out of scope (HR)",
}

# Departments out of scope: their pages are excluded with a recorded reason and
# left out of the department charts.
OUT_OF_SCOPE = {"HR"}

# Decisions ledger shown on the site.
DECISIONS = [
    ("D-13", "Nov 2", "Classify from the record keeper's perspective", "The same page can belong to different departments in different organizations."),
    ("D-15", "Nov 2", "Five-fold cross-validation", "Too few labels per department for a fixed test set."),
    ("D-16", "Nov 5", "Build the categories from Philip Morris's departments", "Ground the labels in how the client filed its records."),
    ("D-27", "Nov 5", "Score at the department level", "Measure whether a record reaches the right office of record. For model scoring, Administration is merged into Legal; the charts of manual labels keep the original departments."),
    ("D-29", "Nov 6", "Run OCR on every pilot page", "Embeddings and scan-quality tests need text for every page."),
    ("D-30", "Nov 6", "Fix page rotation by keeping the more confident reading", "The orientation check alone turned 18% of pages the wrong way."),
    ("D-32", "Nov 6", "Restrict sampling to Philip Morris records", "The benchmark pilot mixed several companies; the taxonomy is Philip Morris's."),
    ("D-33", "Nov 6", "Freeze taxonomy v3: 8 departments, 55 record types", "Capture the finest detail at labeling; types can be merged later."),
    ("D-35", "Nov 6", "Raise the target to 250 if a department projects under 15", "Class balance decides the label count, declared in advance."),
    ("D-37", "Nov 6", "Keep Bates numbers and archive metadata out of the models", "They can reveal the department and inflate scores."),
    ("D-38", "Nov 7", "Label to 250 after the 50-label checkpoint", "Quality, Finance and HR projected under 15 at 200; their handling is decided by counts at the end of labeling."),
    ("D-39", "Nov 7", "Human resources out of scope for this project", "Only 2 of the 262 reviewed documents belonged to it, too few to model; both were excluded with a recorded reason."),
    ("D-40", "Nov 7", "Leave out pages that failed to download", "Two documents in the labeled range returned a web page instead of the scan, so they were never seen. They are recorded as not retrieved, not as reviewed or excluded."),
    ("D-41", "Nov 7", "Train and validate on the 250 labels; test on fresh pages", "Five-fold cross-validation on the labels; 1,000 new documents are filed by the finished model, and a blind-labeled set of them forms the final test (size set by D-48)."),
    ("D-42", "Nov 7", "Establish a pool of 400 pages for labeling", "Three downloads failed, so three replacements were drawn; the 1,000-document scale batch follows them."),
    ("D-43", "Nov 8", "Keep the original PDFs and every archive field", "The first download kept only a reduced image of each first page. All documents were fetched again by ID; the original record is kept and working images are made from it."),
    ("D-44", "Nov 8", "Compare five text sources (four OCR runs and the archive's own text layer) before building the classifier", "The project depends on reading the pages well, and the first engine had been a default nobody compared."),
    ("D-45", "Nov 8", "Group pages by scan quality from the image alone", "Different scan problems need different fixes; the groups were found before any OCR ran."),
    ("D-46", "Nov 8", "Keep the file number in the OCR text, but not in what the classifier reads", "Reading the file number is a real records capability; letting the model use a litigation stamp to decide the department would be a shortcut."),
    ("D-47", "Nov 8", "Accept the sampling rules as they ran", "A code audit checked the draw: it was proportional to the archive, and the duplicate rule keeps every sampled document distinct."),
    ("D-48", "Nov 8", "Final test of 150 blind labels", "A larger test gives a tighter estimate of how well the system files pages it has never seen."),
    ("D-49", "Nov 8", "Test OCR on 22 pages, at least 4 of each kind of scan", "A first check on mostly clean pages flattered one engine; every engine had to be judged on hard pages too."),
    ("D-50", "Nov 8", "Save the position of every word read", "The layout question needs to know where words sit on the page."),
    ("D-51", "Nov 9", "Treat the archive's own text as a reference, not a candidate", "A client's new scans would not come with an archive's text layer, so it shows whether local OCR is worth running but cannot be chosen."),
    ("D-52", "Nov 9", "PaddleOCR is the reading engine", "Chosen by the winning rule set before results: it found the most words correctly on forms with known text, and it is the only top engine that also gives each word's position and a confidence score, at under a second per page."),
    ("D-53", "Nov 9", "Choose the text the classifier reads by how well it files", "Each engine is strongest on different pages, so four versions of the text are compared on filing accuracy and the best one is used."),
    ("D-54", "Nov 10", "Classify each document from its first page", "It matches how every labeled page was judged, from the first page only; whole-document classification is future work."),
    ("D-55", "Nov 12", "Re-fetch the archive's metadata, every field under its own name", "The first download shifted values into the wrong columns (e.g. box); found before any result used it."),
    ("D-56", "Nov 12", "Fix the five folds once: departments balanced, archive boxes kept together", "Every model is scored on the same split, and a page is never tested against a near-copy from its own box."),
    ("D-57", "Nov 12", "The classifier reads the hybrid text", "It scored highest by the rule set in advance, effectively tied with PaddleOCR alone."),
    ("D-58", "Nov 12", "File automatically only at 90% expected accuracy", "A records standard set before any routing result; the cutoff is chosen on training folds only."),
    ("D-59", "Nov 12", "Represent each page by its meaning with a local embedding model", "Word counts missed pages whose purpose differs from their topic; meaning raised macro F1 from 0.56 to 0.68."),
    ("D-60", "Nov 12", "Review suggestions come from a small local language model (Qwen3 8B)", "It runs on the same PC, so pages never leave the machine; one model is kept frozen for the evaluation."),
    ("D-61", "Nov 12", "Judge the suggestion against the classifier's own first choice, with and without its hints", "A rule set before results: the suggestion must be right more often than the classifier on the pages a person reviews."),
    ("D-62", "Nov 12", "Freeze the whole pipeline before the final test", "Every setting was written down before a single test page was predicted, including the 0.55 cutoff (chosen on the pooled cross-validated predictions), so the final numbers could not be tuned to the test."),
    ("D-63", "Nov 12", "Score the 150 held-back pages once and report what fell short", "Auto-filing reached 87% against a 90% target, and meaning tied word counts on new pages; both are findings, and nothing was changed after scoring."),
    ("D-64", "Nov 12", "Run the frozen pipeline on 1,000 new pages", "Answers the client's practical question: how much is filed alone, how much reaches a person, and how long it takes."),
]

# Pipeline stage each decision shaped; the site's file index groups the ledger by it.
DECISION_STAGES = {
    "sample": ["D-32", "D-42", "D-43", "D-47", "D-55"],
    "labels": ["D-13", "D-16", "D-33", "D-35", "D-38", "D-39", "D-40"],
    "ocr": ["D-29", "D-30", "D-44", "D-45", "D-46", "D-49", "D-50", "D-51", "D-52"],
    "classifier": ["D-15", "D-27", "D-37", "D-53", "D-54", "D-56", "D-57", "D-59"],
    "routing": ["D-58", "D-60", "D-61"],
    "final": ["D-41", "D-48", "D-62", "D-63", "D-64"],
}
STAGE_OF = {d: st for st, ids in DECISION_STAGES.items() for d in ids}
_unstaged = [i for i, *_ in DECISIONS if i not in STAGE_OF]
assert not _unstaged, f"Decisions without a stage in DECISION_STAGES: {_unstaged}"

ENGINE_NAMES = {
    "A": "Tesseract, 150 dpi (baseline)", "B": "Tesseract, original resolution",
    "C": "PaddleOCR", "D": "GLM-OCR (vision model)", "E": "Archive's text layer (reference)",
}
GROUP_NAMES = {
    "1": "Clean typed files", "2": "Photographic ads", "3a": "Degraded scans",
    "3b": "Graphics and clippings", "4": "Clean reports, forms, tables",
}


def read_csv(path, **kw):
    """Read a project CSV as strings (cp1252 for files saved by Excel); None if missing."""
    p = ROOT / path
    if not p.exists():
        return None
    try:
        return pd.read_csv(p, dtype=str, keep_default_na=False, **kw)
    except UnicodeDecodeError:
        return pd.read_csv(p, dtype=str, keep_default_na=False, encoding="cp1252", **kw)


# Taxonomy
tax = yaml.safe_load((ROOT / "taxonomy" / "rim_taxonomy_final.yaml").read_text(encoding="utf-8"))
type_info = {}
for f in tax["functions"]:
    for s in f["series"]:
        for t in s["types"]:
            type_info[t["id"]] = (f["id"], s["name"], t["name"])

# Labels (all annotators, newest answer per document)
labels = lm.load_labels()
headline = lm.summary(labels)
valid = labels[labels["outcome"] == "labeled"]
n_valid = len(valid)

by_dept = []
for dep_id, name in DEPT_NAMES.items():
    if dep_id in OUT_OF_SCOPE:
        continue
    d = valid[valid["function_id"] == dep_id]
    unsure = int((d["certainty"] == "unsure").sum())
    by_dept.append({
        "id": dep_id, "name": name, "certain": len(d) - unsure, "uncertain": unsure,
        "projected": round(len(d) * TARGET / n_valid) if n_valid else 0,
    })

def outcomes_of(df):
    """Labeled certain / uncertain / no fit / excluded by reason, for any set of labels."""
    v = df[df["outcome"] == "labeled"]
    ex = df[df["outcome"] == "exclude"]
    counts = {lab: 0 for lab in EXCLUDE_REASONS.values()}
    other = 0
    for note in ex["note"]:
        hit = next((lab for key, lab in EXCLUDE_REASONS.items() if note.lower().startswith(key)), None)
        if hit:
            counts[hit] += 1
        else:
            other += 1
    out = [{"label": "Labeled, certain", "count": int((v["certainty"] != "unsure").sum())},
           {"label": "Labeled, uncertain", "count": int((v["certainty"] == "unsure").sum())},
           {"label": "No fit", "count": int((df["outcome"] == "no_fit").sum())}]
    out += [{"label": f"Excluded: {k[0].lower() + k[1:]}", "count": n} for k, n in counts.items() if n]
    if other:
        out.append({"label": "Excluded: other", "count": other})
    return out


def pairs_of(df):
    """Department pairs recorded as primary and secondary on the same page."""
    v = df[(df["outcome"] == "labeled") & (df["also_fits"] != "")]
    found = {}
    for _, r in v.iterrows():
        a = SHORT.get(r["function_id"], r["function_id"])
        b = SHORT.get(type_info.get(r["also_fits"], (r["also_fits"],))[0], "?")
        if a != b:
            key = tuple(sorted((a, b)))
            found[key] = found.get(key, 0) + 1
    ranked = sorted(found.items(), key=lambda x: -x[1])
    cut = ranked[7][1] if len(ranked) > 8 else 0          # top 8, plus any pair tied with the 8th
    return [{"a": a, "b": b, "count": c} for (a, b), c in ranked if c >= cut]


def type_counts_of(df):
    """Return labeled pages per record type."""
    return {k: int(n) for k, n in df.loc[df["outcome"] == "labeled", "type_id"].value_counts().items()}


outcomes = outcomes_of(labels)
pairs = pairs_of(labels)
type_counts = type_counts_of(labels)
tree = []
for f in tax["functions"]:
    tree.append({"name": f["name"], "series": [
        {"name": s["name"], "types": [{"id": t["id"], "name": t["name"], "count": int(type_counts.get(t["id"], 0))}
                                      for t in s["types"]]} for s in f["series"]]})
tree.sort(key=lambda d: list(DEPT_NAMES.values()).index(d["name"]) if d["name"] in DEPT_NAMES.values() else 99)

progress = []
if n_valid:
    by_day = valid["saved_at"].str[:10].value_counts().sort_index().cumsum()
    progress = [{"date": d, "valid": int(v)} for d, v in by_day.items()]

manifest = read_csv("data/pm_sample/manifest.csv")
drawn = len(manifest) if manifest is not None else 0

# Progress by batch of 100 (manifest order = labeling order)
batches = []
skips = lm.load_skips(labels)
skipped_paths = set(skips["path"])
if manifest is not None:
    status = labels.set_index("path")
    for b in range(0, drawn, 100):
        paths = manifest["image"].iloc[b:b + 100]
        st = status.reindex(paths)
        lab = st[st["outcome"] == "labeled"]
        batches.append({
            "batch": f"Batch {b // 100 + 1}",
            "certain": int((lab["certainty"] != "unsure").sum()),
            "uncertain": int((lab["certainty"] == "unsure").sum()),
            "set_aside": int(st["outcome"].isin(["exclude", "no_fit"]).sum()),
            # Skipped (e.g. failed download): waiting to be labeled, shown apart
            # from "to go" so a finished batch doesn't look unfinished.
            "skipped": int(paths.isin(skipped_paths).sum()),
            "remaining": int((st["outcome"].isna() & ~paths.isin(skipped_paths).values).sum()),
        })
# Per-batch view: same shared counting, one entry per batch with its headline
# numbers and department counts.
by_batch = []
if manifest is not None and len(labels):
    lb = labels.assign(batch=lm.batch_of(labels["path"], manifest))
    sk = skips.assign(batch=lm.batch_of(skips["path"], manifest))
    for b in sorted(lb["batch"].dropna().unique()):
        g = lb[lb["batch"] == b]
        head = lm.summary(g)
        gv = g[g["outcome"] == "labeled"]
        depts = []
        for dep_id, name in DEPT_NAMES.items():
            if dep_id in OUT_OF_SCOPE:
                continue
            d = gv[gv["function_id"] == dep_id]
            u = int((d["certainty"] == "unsure").sum())
            depts.append({"id": dep_id, "name": name, "certain": len(d) - u, "uncertain": u})
        head["unsure_share"] = round(float((gv["certainty"] == "unsure").mean()), 3) if len(gv) else None   # share of valid labels
        by_batch.append({"batch": f"Batch {int(b)}", **{k: head[k] for k in
                         ("reviewed", "valid", "excluded", "unsure", "unsure_share")},
                         "skipped": int((sk["batch"] == b).sum()), "by_department": depts,
                         "outcomes": outcomes_of(g), "pairs": pairs_of(g),
                         "type_counts": type_counts_of(g)})

# Did the department mix settle? Each department's share of valid labels
# as labeling went on, in the order documents were first labeled.
first_saved = lm.all_saves().groupby("path")["saved_at"].min()
order = valid.assign(first=valid["path"].map(first_saved)).sort_values("first")
settling = {"steps": [], "series": {d: [] for d in DEPT_NAMES if d not in OUT_OF_SCOPE}, "batch_ends": []}
for n in list(range(20, len(order), 5)) + ([len(order)] if len(order) >= 20 else []):
    head_n = order.iloc[:n]["function_id"].value_counts()
    settling["steps"].append(n)
    for dep_id in settling["series"]:
        settling["series"][dep_id].append(round(float(head_n.get(dep_id, 0)) / n, 3))
if manifest is not None and len(order):
    ob = lm.batch_of(order["path"], manifest)
    for b in sorted(ob.dropna().unique())[:-1]:          # where each finished batch ended
        settling["batch_ends"].append(int((ob <= b).sum()))

# Documents needed to reach the target at the current set-aside rate
set_aside_rate = (len(labels) - n_valid) / len(labels) if len(labels) else 0
needed = round(TARGET / (1 - set_aside_rate)) if set_aside_rate < 1 else None


# ---------- Pilot rounds (taxonomy evidence) ----------
def skim(path, label, fallback, current=False):
    """Return a pilot round's counts, or the fallback when its labels are missing."""
    df = read_csv(path)
    if df is not None and len(df):
        df = df.drop_duplicates("path", keep="last")
        # Valid = reviewed and not excluded (no fit counts as excluded); the uncertain
        # share is unsure valid labels / valid labels, as in the final round.
        v = df[df["outcome"] == "labeled"]
        return {"label": label, "pages": len(df), "reviewed": len(df), "valid": len(v),
                "no_fit": int((df["outcome"] == "no_fit").sum()), "excluded": len(df) - len(v),
                "unsure": round(float((v["certainty"] == "unsure").mean()), 3) if len(v) else None,
                "current": current}
    return {"label": label, **fallback, "current": current}


skims = [skim("labels/skim_labels.csv", "Generic categories", {"pages": 64, "unsure": .42}),
         skim("labels/skim_v2_labels.csv", "Philip Morris departments", {"pages": 100, "unsure": .24}, True)]

# OCR
funsd = {"forms": 50, "words": 8726, "wer": .369, "cer": .263, "recall": .747, "spearman": .63, "points": []}
fe = read_csv("results/funsd_ocr_eval.csv")
if fe is not None and len(fe):
    fe = fe.astype({"mean_conf": float, "word_recall": float, "true_words": int})
    funsd["forms"] = len(fe)
    funsd["words"] = int(fe["true_words"].sum())
    funsd["recall"] = round(float((fe["word_recall"] * fe["true_words"]).sum() / fe["true_words"].sum()), 3)
    funsd["spearman"] = round(float(fe["mean_conf"].rank().corr(fe["word_recall"].rank())), 2)
    funsd["points"] = [{"conf": round(c, 1), "recall": round(r, 3)} for c, r in zip(fe["mean_conf"], fe["word_recall"])]

spot = []
sc = read_csv("results/rvl_pilot/ocr_spotcheck.csv")
if sc is not None:
    for _, r in sc.iterrows():
        if r.get("rating"):
            spot.append({"page_type": r["rvl_class"].capitalize(), "conf": float(r["mean_conf"]), "rating": int(float(r["rating"]))})
    spot.sort(key=lambda r: r["conf"])

pilot = []
oc = read_csv("results/rvl_pilot/ocr_results.csv")
if oc is not None and "mean_conf" in oc:
    oc["type"] = oc["path"].str.split("/").str[0].str.capitalize()
    oc["mean_conf"] = pd.to_numeric(oc["mean_conf"], errors="coerce")
    pilot = [{"type": t, "conf": round(float(c), 1)}
             for t, c in oc.groupby("type")["mean_conf"].mean().sort_values(ascending=False).items()]
else:
    pilot = [{"type": t, "conf": c} for t, c in [
        ("Resume", 88.9), ("Email", 87.5), ("Memo", 84.7), ("Letter", 84.4), ("Scientific publication", 83.1),
        ("Presentation", 81.6), ("News article", 79.9), ("Scientific report", 79.3), ("Questionnaire", 78.3),
        ("Form", 72.8), ("Specification", 72.1), ("Budget", 70.8), ("Invoice", 64.5), ("Advertisement", 63.5),
        ("File folder", 52.2), ("Handwritten", 37.8)]]

# OCR engine comparison
def records(path):
    """A comparison table as a list of rows with real numbers (not strings); [] if missing."""
    p = ROOT / path
    if not p.exists():
        return []
    df = pd.read_csv(p, dtype={"group": str, "assigned_group": str})
    return json.loads(df.to_json(orient="records"))


eda = {}
for engine in ENGINE_NAMES:
    p = ROOT / "results" / "ocr_eda" / engine / "summary.json"
    if p.exists():
        eda[engine] = json.loads(p.read_text(encoding="utf-8"))

scan_groups = []
profile = read_csv("data/pm_sample/page_profile.csv")
if profile is not None and "group" in profile:
    counts = profile["group"].value_counts()
    scan_groups = [{"id": g, "name": n, "pages": int(counts.get(g, 0))} for g, n in GROUP_NAMES.items()]

orientation = None
paddle = read_csv("data/pm_sample/ocr_paddle/ocr_results.csv")
if paddle is not None and "suggested_angle" in paddle:
    angle = pd.to_numeric(paddle["suggested_angle"], errors="coerce").fillna(0)
    rotated = pd.to_numeric(paddle["rotated"], errors="coerce").fillna(0)
    orientation = {"pages": len(paddle), "turn_suggested": int((angle != 0).sum()),
                   "turn_kept": int((rotated != 0).sum())}

# How many scan groups the measures support (checked afterwards). k = 4 was set as a
# starting point; one group was later split by a speckle rule (the final groups).
k_check = None
kc = read_csv("results/page_groups_k_check.csv")
if kc is not None and {"k", "inertia", "silhouette", "group_sizes", "used"} <= set(kc.columns):
    def _sizes(v):
        return [int(x) for x in str(v).replace(" ", "").split("/") if x.isdigit()]
    tried = kc[kc["k"].str.isdigit()]
    final = kc[kc["k"] == "final"]
    k_check = {
        "tried": [{"k": int(r["k"]), "inertia": float(r["inertia"]), "silhouette": float(r["silhouette"]),
                   "sizes": _sizes(r["group_sizes"]), "used": r["used"].strip().lower() == "yes"}
                  for _, r in tried.iterrows()],
        "final": ({"groups": len(_sizes(final.iloc[0]["group_sizes"])), "silhouette": float(final.iloc[0]["silhouette"]),
                   "sizes": _sizes(final.iloc[0]["group_sizes"])} if len(final) else None),
    }

text_options = None
sources = read_csv("data/pm_sample/ocr_text/sources.csv")
if sources is not None:
    text_options = {"options": ["C", "hybrid", "both", "D"],
                    "pages": len(sources),
                    "hybrid_from_D": int((sources["hybrid_source"] == "D").sum()),
                    "no_D_reading": int((sources["d_available"] == "False").sum())}

engines = {
    "names": ENGINE_NAMES,
    "backbone": "C",                      # main engine
    "reference": ["E"],                   # reference only
    "analyses": eda,                      # per engine, all 1,400 pages
    "pages_by_group": records("results/ocr_compare/pages_by_group.csv"),
    "review_overall": records("results/ocr_compare/review_overall.csv"),
    "review_by_group": records("results/ocr_compare/review_by_group.csv"),
    "funsd": records("results/ocr_compare/funsd.csv"),
    "group_check": records("results/ocr_compare/group_check.csv"),
    "scan_groups": scan_groups,
    "orientation": orientation,
    "text_options": text_options,
}
ocr_done = bool(engines["funsd"]) and bool(engines["review_overall"])

# Classifier
# Cross-validated results from train_tfidf.py, bates_probe.py, train_embeddings.py,
# review_routing.py and tfidf_errors.py. Counts and scores only; no page IDs.
CLS = "results/classifier"
FINAL_TEST_PAGES = 150   # held back, labeled blind, scored once at the end


def num_csv(path, **kw):
    """Read a result table with numeric columns; None if missing."""
    p = ROOT / path
    return pd.read_csv(p, **kw) if p.exists() else None


def run_stats(df):
    """Return mean and spread of macro F1, mean accuracy and per-fold macro F1."""
    return {"macro_f1": round(float(df["macro_f1"].mean()), 3),
            "macro_f1_sd": round(float(df["macro_f1"].std()), 3),
            "accuracy": round(float(df["accuracy"].mean()), 3),
            "folds": [round(float(v), 3) for v in df["macro_f1"]]}


def per_department(path):
    """Return F1 per department from a classification report table."""
    df = num_csv(path, index_col=0)
    return None if df is None else {str(k): round(float(r["f1-score"]), 3) for k, r in df.iterrows()}


classifier = None
tf_cv = num_csv(f"{CLS}/tfidf_cv_results.csv")
emb_cv = num_csv(f"{CLS}/emb_cv_results.csv")
emb_pred = num_csv(f"{CLS}/emb_cv_predictions.csv", dtype={"id": str})
if tf_cv is not None and emb_cv is not None and emb_pred is not None:
    tfidf_runs = {r: run_stats(g) for r, g in tf_cv.groupby("run")}
    emb_runs = {"hybrid": run_stats(emb_cv)}
    for name in ("hybrid_with_bates", "A"):
        extra = num_csv(f"{CLS}/emb_{name}_cv_results.csv")
        if extra is not None:
            emb_runs[name] = run_stats(extra)
    probe = num_csv(f"{CLS}/bates_probe.csv", dtype={"fold": str})
    probe_stats = run_stats(probe[probe["fold"] != "mean"]) if probe is not None else None

    confusion = None
    conf = num_csv(f"{CLS}/confusion_embeddings.csv", index_col=0)
    if conf is not None:
        conf = conf.drop(index="All", errors="ignore").drop(columns="All", errors="ignore")
        confusion = {"labels": [str(x) for x in conf.index], "guesses": [str(x) for x in conf.columns],
                     "counts": conf.values.astype(int).tolist()}

    lab = pd.read_csv(ROOT / "labels" / "label_pm_labels_salo.csv", dtype=str)
    lab = lab.sort_values("saved_at").drop_duplicates("doc_id", keep="last")
    certainty_of = dict(zip(lab["doc_id"], lab["certainty"]))

    def by_certainty(pred, col):
        pred = pred.assign(c=pred["id"].map(certainty_of), ok=pred["true"] == pred[col])
        return {k: {"pages": int(len(g)), "accuracy": round(float(g["ok"].mean()), 3)} for k, g in pred.groupby("c")}

    tf_pred = num_csv(f"{CLS}/tfidf_cv_predictions.csv", dtype={"id": str})
    certainty = {"embeddings": by_certainty(emb_pred, "predicted_raw")}
    if tf_pred is not None:
        certainty["tfidf"] = by_certainty(tf_pred[tf_pred["run"] == "hybrid"], "predicted")

    def curve(path, model=None):
        df = num_csv(path)
        if df is None:
            return []
        if model:
            df = df[df["model"] == model].drop(columns="model")
        return json.loads(df.to_json(orient="records"))

    auto = emb_pred["routed"] == "auto-filed"
    correct = emb_pred["correct"].astype(str) == "True"
    units = (pd.DataFrame({"a": auto.astype(int), "c": correct.astype(int)})
             .sort_values(["a", "c"], ascending=[False, False]).values.tolist())
    dev_ids = read_csv("results/cv_folds.csv")["id"]
    txt = ROOT / "data" / "pm_sample" / "ocr_text_clean"
    read_txt = lambda opt, i: (txt / opt / f"{i}.txt").read_text(encoding="utf-8") if (txt / opt / f"{i}.txt").exists() else None
    tf_pred = read_csv(f"{CLS}/tfidf_cv_predictions.csv")
    tp = tf_pred.pivot_table(index="id", columns="run", values="predicted", aggfunc="first")
    hybrid_vs_paddle = {"pages": int(len(dev_ids)),
                        "text_differs": int(sum(read_txt("hybrid", i) != read_txt("C", i) for i in dev_ids)),
                        "prediction_differs": int((tp["hybrid"] != tp["C"]).sum())}
    classifier = {
        "pages": int(len(emb_pred)), "folds": 5, "final_test_pages": FINAL_TEST_PAGES,
        "hybrid_vs_paddle": hybrid_vs_paddle,
        "embedding": {"model": "Qwen3-Embedding-0.6B",
                      "size": int(np.load(ROOT / "results" / "embeddings" / "hybrid_qwen3-0.6b.npy", mmap_mode="r").shape[1])},
        "tfidf": tfidf_runs, "embeddings": emb_runs, "bates_probe": probe_stats,
        "per_department": {"tfidf": per_department(f"{CLS}/per_department_hybrid.csv"),
                           "embeddings": per_department(f"{CLS}/per_department_embeddings.csv")},
        "confusion": confusion, "certainty": certainty,
        "routing": {"target": 0.9, "pages": int(len(emb_pred)), "auto_filed": int(auto.sum()),
                    "auto_correct": int((auto & correct).sum()),
                    "accuracy_auto_filed": round(float(correct[auto].mean()), 3) if auto.any() else None,
                    "by_fold": json.loads(emb_cv[["fold", "threshold_from_training", "auto_filed", "coverage",
                                                  "accuracy_auto_filed"]].to_json(orient="records")),
                    "curve_embeddings": curve(f"{CLS}/emb_routing_curve.csv"),
                    "curve_tfidf": curve(f"{CLS}/routing_curve.csv", "calibrated"),
                    "units": units},
    }


# Record-type suggestions, final test, scale run
# Counts and rates only: no page IDs, page text or model reasons reach the site.
from math import comb   # noqa: E402


def mcnemar_exact(a, b):
    """Two-sided exact McNemar p (binomial test on the discordant pairs)."""
    n, k = a + b, min(a, b)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)


def paired(right_a, right_b):
    """Pages only A got right, only B got right, exact McNemar p."""
    a, b = int((right_a & ~right_b).sum()), int((~right_a & right_b).sum())
    return {"only_rag": a, "only_classifier": b,
            "p": round(mcnemar_exact(a, b), 4) if a + b else None}


def jsonl(path):
    """Read a JSON-lines file into a data frame; None if missing."""
    p = ROOT / path
    return pd.DataFrame([json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]) if p.exists() else None


rag = None
ret = num_csv("results/rag/retrieval_summary.csv")
ragA, ragC = num_csv("results/rag/rag_summary_A.csv"), num_csv("results/rag/rag_summary_C.csv")
sugA, sugC = jsonl("results/rag/suggestions_A.jsonl"), jsonl("results/rag/suggestions_C.jsonl")
if ret is not None and ragA is not None and sugA is not None:
    q = ret[ret["pages"] == "review queue"].iloc[0]
    approaches = [
        {"name": "Similar filed pages vote", "top1": float(q["neighbors_top1"]), "top2": float(q["neighbors_top2"])},
        {"name": "Classifier", "top1": float(q["classifier_top1"]), "top2": float(q["classifier_top2"])},
    ]
    if ragC is not None:
        approaches.append({"name": "Language model, no classifier hints",
                           "top1": float(ragC["rag_top1"][0]), "top2": float(ragC["rag_top2"][0])})
    approaches.append({"name": "RAG suggestion (with classifier's top two)",
                       "top1": float(ragA["rag_top1"][0]), "top2": float(ragA["rag_top2"][0])})
    rag = {"review_pages": int(ragA["pages"][0]), "approaches": approaches,
           "paired_A": paired(sugA["department"] == sugA["true"], sugA["classifier_first"] == sugA["true"]),
           "paired_C": (paired(sugC["department"] == sugC["true"], sugC["classifier_first"] == sugC["true"])
                        if sugC is not None else None),
           "changed_answer": int((sugA["department"] != sugA["classifier_first"]).sum()),
           "seconds_per_page": float(ragA["seconds_per_page"][0]),
           "model": "Qwen3 8B (local, Ollama)", "neighbors": 5}

final_test = None
FT = "results/final_test"
ft_scores = num_csv(f"{FT}/scores.csv")
if ft_scores is not None:
    sc = {r["measure"]: {"value": r["value"], "low": None if pd.isna(r["ci_low"]) else r["ci_low"],
                         "high": None if pd.isna(r["ci_high"]) else r["ci_high"]} for _, r in ft_scores.iterrows()}
    ft_pages = num_csv(f"{FT}/scored_pages.csv", dtype={"id": str})
    cmp_ = num_csv(f"{FT}/rag_comparison.csv").iloc[0]
    ft_excl = num_csv(f"{FT}/excluded_check.csv", dtype={"id": str})
    a = (ft_pages["routed"] == "auto-filed").astype(int)
    c = (ft_pages["correct"].astype(str) == "True").astype(int)
    ft_units = pd.DataFrame({"a": a, "c": c}).sort_values(["a", "c"], ascending=[False, False]).values.tolist()
    final_test = {
        "pages": int(sc["pages_scored"]["value"]), "cutoff": 0.55, "target": 0.9,
        "scores": {k: v for k, v in sc.items() if k not in ("pages_scored", "auto_filed_pages")},
        "auto_filed": int(sc["auto_filed_pages"]["value"]), "auto_correct": int((a & c).sum()),
        "per_department": json.loads(num_csv(f"{FT}/per_department.csv").to_json(orient="records")),
        "rag": {"review_pages": int(cmp_["review_pages"]),
                "classifier_top1": float(cmp_["classifier_top1"]), "rag_top1": float(cmp_["rag_top1"]),
                "classifier_top2": float(cmp_["classifier_top2"]), "rag_top2": float(cmp_["rag_top2"]),
                "only_rag": int(cmp_["rag_only_right"]), "only_classifier": int(cmp_["classifier_only_right"]),
                "p": float(cmp_["mcnemar_exact_p"])},
        "certainty": json.loads(num_csv(f"{FT}/certainty.csv").to_json(orient="records")),
        "curve": json.loads(num_csv(f"{FT}/routing_curve.csv").to_json(orient="records")),
        "excluded": {"pages": int(len(ft_excl)), "to_a_person": int((ft_excl["routed"] == "review").sum()),
                     "by_token_rule": int((ft_excl["route_reason"] == "under 20 tokens").sum())},
        "units": ft_units,
        "confidence_quartiles": [round(float(q), 3) for q in ft_pages["confidence"].astype(float).quantile([.25, .5, .75])],
    }
    ftl = read_csv("labels/final_test_labels_salo.csv")
    if ftl is not None and len(ftl):                      # date of the newest final-test label
        final_test["labels_as_of"] = datetime.fromisoformat(ftl["saved_at"].max()).strftime("%b %d, %Y").replace(" 0", " ")

scale_run = None
SR = "results/scale_run"
srs = num_csv(f"{SR}/summary.csv")
if srs is not None:
    v = dict(zip(srs["measure"], srs["value"]))
    num = lambda k: float(v[k])
    sr_timing = num_csv(f"{SR}/timing.csv")
    sr_spot = num_csv(f"{SR}/spotcheck_result.csv")
    sr_spot_lab = (ROOT / "labels" / "spotcheck_salo.csv")
    sr_spot_excluded = 0
    if sr_spot_lab.exists():
        sl = pd.read_csv(sr_spot_lab, dtype=str).sort_values("saved_at").drop_duplicates("doc_id", keep="last")
        sr_spot_excluded = int((sl["outcome"] != "labeled").sum())
    scale_run = {
        "pages": int(num("pages")), "auto_filed": int(num("auto_filed")), "to_a_person": int(num("to_a_person")),
        "reasons": {"below cutoff": int(num("reason_below_cutoff")), "under 20 tokens": int(num("reason_under_20_tokens")),
                    "no readable text": int(num("reason_no_readable_text"))},
        "projected_wrong": {"value": num("projected_wrong_auto_filed"), "low": num("projected_wrong_low"),
                            "high": num("projected_wrong_high")},
        "rag_agrees": num("rag_agrees_with_classifier_first"),
        "timing": [{"stage": r["stage"], "seconds": float(r["seconds_per_page"])}
                   for _, r in sr_timing.iterrows() if r["stage"] != "Total"],
        "departments": json.loads(num_csv(f"{SR}/departments.csv").to_json(orient="records")),
        "spotcheck": ({"checked": int(len(sr_spot)), "agree": int(sr_spot["agrees"].astype(str).eq("True").sum()),
                       "excluded": sr_spot_excluded} if sr_spot is not None else None),
        "confidence_quartiles": [round(float(q), 3) for q in
                                 num_csv(f"{SR}/predictions.csv")["confidence"].astype(float).quantile([.25, .5, .75])],
    }

# Pipeline status strip
# "Human review" means a person accepting or correcting suggestions in the app,
# with each decision logged. REVIEW_LOG is unused: the strip marks every stage done.
REVIEW_LOG = None
review_logged = REVIEW_LOG is not None and Path(REVIEW_LOG).exists()
# The stage strip: every stage is complete; numbers from the data and the taxonomy.
_emb, _tf = final_test["scores"]["macro_f1"]["value"], final_test["scores"]["tfidf_macro_f1"]["value"]
pipeline = [
    {"step": "Sample", "status": "done", "value": f"{drawn:,} documents randomly sampled"},
    {"step": "Filing system", "status": "done",
     "value": f"{len(tax['functions'])} departments, {sum(len(f['series']) for f in tax['functions'])} series, "
              f"{len(type_info)} record types"},
    {"step": "Labels", "status": "done", "value": f"{n_valid} development labels"},
    {"step": "Reading / OCR", "status": "done", "value": "Five text sources compared; PaddleOCR selected"},
    {"step": "Classifier", "status": "done",
     "value": (f"Sentence embeddings and TF-IDF both scored macro F1 {_emb:.2f} on the final test" if f"{_emb:.2f}" == f"{_tf:.2f}"
               else f"Macro F1 on the final test: {_emb:.2f} with sentence embeddings, {_tf:.2f} with TF-IDF")},
    {"step": "Routing and review", "status": "done",
     "value": f"{final_test['auto_filed'] / final_test['pages']:.0%} qualified for automatic filing; "
              "the remaining test pages were routed for review"},
]

# Findings
# Record-type depth on the final test: the pipeline's suggestion scored on all 150 pages
# (score_record_types.py; Administration counted with Legal).
rt = num_csv("results/record_types/summary.csv")
if final_test is not None and rt is not None:
    row = rt[(rt["set"] == "test") & (rt["route"] == "all")].iloc[0]
    final_test["record_type_depth"] = {"pages": int(row["pages"]), "department": round(float(row["department_scored"]), 3),
                                       "series": round(float(row["series"]), 3), "record_type": round(float(row["record_type"]), 3),
                                       "record_type_or_also_fits": round(float(row["record_type_or_also_fits"]), 3)}
# Automatically filed final-test pages with little text (the 20-token rule's margin): 60 tokens or fewer.
if final_test is not None:
    THIN = 60
    fp = num_csv(f"{FT}/predictions.csv", dtype={"id": str})
    sp = num_csv(f"{FT}/scored_pages.csv", dtype={"id": str}).set_index("id")
    thin = fp[(fp["routed"] == "auto-filed") & (fp["tokens"].astype(float) <= THIN)]
    wrong = [i for i in thin["id"] if i in sp.index and str(sp.loc[i, "correct"]) != "True"]
    # Test pages sent to review by the 20-token rule (valid and excluded)
    under = fp[fp["route_reason"] == "under 20 tokens"]
    final_test["under_20_tokens"] = {"pages": int(len(under)), "of": int(len(fp)),
                                     "valid": int(under["id"].isin(sp.index).sum()),
                                     "excluded": int((~under["id"].isin(sp.index)).sum())}
    final_test["thin_auto_filed"] = {"max_tokens": THIN, "pages": int(len(thin)),
                                     "shortest": int(thin["tokens"].astype(float).min()) if len(thin) else None,
                                     "wrong": len(wrong)}
# The archive assistant's check (two rounds; the second with blind questions).
archive_check = None
ac1, ac2 = num_csv("results/archive_check/archive_check_summary.csv"), num_csv("results/archive_check/archive_check_r2_summary.csv")
if ac1 is not None and ac2 is not None:
    r1 = ac1.iloc[0]
    g = ac2.set_index("group")
    rounds = [{"answerable": int(r1["answerable_questions"]), "cited": int(r1["cited_the_known_record"]),
               "out_of_archive": int(r1["out_of_archive_questions"]), "declined": int(r1["correctly_declined"])},
              {"answerable": int(g.loc["all answerable", "questions"]), "cited": int(g.loc["all answerable", "cited_a_relevant_record"]),
               "out_of_archive": int(g.loc["not in archive", "questions"]), "declined": int(g.loc["not in archive", "correctly_declined"])}]
    # Each round's median seconds, and the faithfulness judgment of each answer against its cited pages
    rounds[0]["median_seconds"] = round(float(r1["median_seconds"]), 1)
    rounds[1]["median_seconds"] = round(float(g.loc["time", "median_seconds"]), 1)
    archive_check = {"rounds": rounds, **{k: sum(r[k] for r in rounds) for k in ("answerable", "cited", "out_of_archive", "declined")}}
    acq1, acq2 = read_csv("results/archive_check/archive_check.csv"), read_csv("results/archive_check/archive_check_r2.csv")
    if acq1 is not None and acq2 is not None:
        f = pd.concat([acq1[acq1["known_file_id"] != ""]["faithful"], acq2[acq2["kind"] != "not in archive"]["faithful"]])
        archive_check["support"] = {"fully": int((f == "Yes").sum()), "partly": int((f == "Partly").sum()),
                                    "not": int((f == "No").sum()), "declined": int(f.isin(["", "n/a"]).sum())}   # declined answers are marked n/a

# The application's Archive (app page): filed records as the app shows them = archive.csv
# without the five pages flagged for personal data (localrim.intake.HIDDEN) + records filed in the app.
app_archive = None
arc = read_csv("outputs/app/archive.csv")
if arc is not None:
    from localrim.intake import HIDDEN
    filed_app = read_csv("outputs/app/filed.csv")
    n_filed = 0 if filed_app is None else int((~filed_app.get("id", pd.Series(dtype=str)).isin(HIDDEN)).sum())
    app_archive = {"records": int((~arc["id"].isin(HIDDEN)).sum()) + n_filed,
                   "departments": int(arc.loc[~arc["id"].isin(HIDDEN), "department"].nunique())}

data = {
    "updated": datetime.now().strftime("%b %d, %Y %H:%M").replace(" 0", " "),
    # Time of the newest saved label: says exactly which records the site shows.
    "data_as_of": (datetime.fromisoformat(headline["data_as_of"]).strftime("%b %d, %Y %H:%M").replace(" 0", " ")
                   if headline["data_as_of"] else None),
    "deadline": DEADLINE,
    "pipeline": pipeline,
    "sample": {"drawn": drawn, "labeling_pool": LABEL_POOL, "funnel": [
        {"label": "Philip Morris Records", "count": PM_TOTAL},
        {"label": "Public", "count": PM_PUBLIC},
        {"label": "Public, scanned paper", "count": PM_PUBLIC_SCANNED},
        {"label": "Drawn at random", "count": drawn},
        {"label": "Reviewed", "count": len(labels)},
        {"label": "Valid labels", "count": n_valid},
    ]},
    "taxonomy": {"departments": len(tax["functions"]),
                 "series": sum(len(f["series"]) for f in tax["functions"]),
                 "types": len(type_info), "skims": skims, "tree": tree},
    "labels": {"target": TARGET, "valid": n_valid, "reviewed": len(labels),
               # unsure valid labels / valid labels (the batch records keep unsure / reviewed,
               # localrim_metrics' definition)
               "uncertain_share": round(float((valid["certainty"] == "unsure").mean()), 3) if n_valid else None,
               "low_count_rule": LOW_COUNT_RULE, "by_department": by_dept, "outcomes": outcomes,
               "pairs": pairs, "progress": progress, "batches": batches,
               "set_aside_rate": round(set_aside_rate, 3), "needed": needed,
               "skipped": len(skipped_paths), "excluded": headline["excluded"], "by_batch": by_batch,
               "type_counts": type_counts, "settling": settling},
    "ocr": {"funsd": funsd, "spotcheck": spot, "pilot_by_type": pilot, "engines": engines, "k_check": k_check},
    "classifier": classifier,
    "rag": rag,
    "final_test": final_test,
    "scale_run": scale_run,
    "archive_check": archive_check,
    "app_archive": app_archive,
    "decisions": [{"id": i, "date": d, "decision": t, "why": w, "stage": STAGE_OF[i]} for i, d, t, w in DECISIONS],
}

# Consistency check
# Write the batch records from the same labels, then make sure the site and the
# batch records agree before anything is written. Stop if they don't.
rows = batch_metrics.write(SNAP_DIR / f"batch_metrics_{SNAPSHOT}.csv" if SNAPSHOT else None)
checks = {
    "reviewed": (len(labels), sum(r["reviewed"] for r in rows)),
    "valid": (n_valid, sum(r["valid"] for r in rows)),
    "valid (site batches)": (n_valid, sum(b["certain"] + b["uncertain"] for b in batches)),
    "unsure": (headline["unsure"], sum(r["unsure"] for r in rows)),
    "skipped": (len(skipped_paths), sum(r["skipped"] for r in rows)),
    "skipped (site batches)": (len(skipped_paths), sum(b["skipped"] for b in batches)),
    "valid (batch view)": (n_valid, sum(b["valid"] for b in by_batch)),
    "outcomes (batch view)": (sum(o["count"] for o in outcomes),
                              sum(o["count"] for b in by_batch for o in b["outcomes"])),
    "record types (batch view)": (sum(type_counts.values()),
                                  sum(n for b in by_batch for n in b["type_counts"].values())),
}
bad = {k: v for k, v in checks.items() if v[0] != v[1]}
if bad:
    sys.exit("Site and batch records disagree, nothing written: " +
             "; ".join(f"{k}: site {a} vs batches {b}" for k, (a, b) in bad.items()))

OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(data, indent=1), encoding="utf-8")
print(f"Wrote {OUT.relative_to(ROOT)}: {len(labels)} reviewed, {n_valid} valid labels, "
      f"{data['labels']['uncertain_share']:.0%} of valid labels unsure, data as of {data['data_as_of']} (matches batch records)")
