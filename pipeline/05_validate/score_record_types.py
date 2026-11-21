"""Score the record-type suggestions at every level of the taxonomy.

Added after the pipeline was frozen; nothing is tuned on these numbers.

Pages:
    training  the 250 training pages, cross-validated (review pages from
              results/rag/suggestions_A.jsonl, auto-filed from suggestions_A_autofiled.jsonl)
    test      the 150 final-test pages (review pages from results/final_test/suggestions_A.jsonl,
              auto-filed from results/scale_run/suggestions_A_autofiled.jsonl)
Levels: department, department_scored (Administration counted with Legal), series,
record_type and record_type_top2. Each is also scored "_or_also_fits": right when the first
choice matches the label or its "also fits" choice, reported beside the strict score.

Writes results/record_types/summary.csv (set x route), by_department.csv and per_page.csv.

Usage:
    python pipeline/05_validate/score_record_types.py
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "record_types"
TAXONOMY = ROOT / "taxonomy" / "rim_taxonomy_final.yaml"
MERGE = {"ADM": "LEG"}  # Administration is scored with Legal


def latest(path: Path) -> pd.DataFrame:
    """Read a labels file, keeping each document's newest answer."""
    df = pd.read_csv(path, dtype=str).fillna("")
    return df.sort_values("saved_at").drop_duplicates("doc_id", keep="last")


def read_jsonl(path: Path, route: str) -> pd.DataFrame:
    """Read a JSON-lines file into a data frame."""
    rows = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    df = pd.DataFrame(rows)[["id", "record_type", "second_record_type", "reason"]]
    df["route"] = route
    return df


def main() -> None:
    """Score every page at every level and save the tables."""
    OUT.mkdir(parents=True, exist_ok=True)
    tax = yaml.safe_load(TAXONOMY.read_text(encoding="utf-8"))
    series_of = {t["id"]: s["id"] for f in tax["functions"] for s in f["series"] for t in s["types"]}

    train_ids = set(pd.read_csv(ROOT / "results" / "cv_folds.csv", dtype=str)["id"])
    train_lab = latest(ROOT / "labels" / "label_pm_labels_salo.csv")
    train_lab = train_lab[train_lab["doc_id"].isin(train_ids)]
    test_lab = latest(ROOT / "labels" / "final_test_labels_salo.csv")
    test_lab = test_lab[test_lab["outcome"] == "labeled"]

    sets = {
        "training": (train_lab, pd.concat([
            read_jsonl(ROOT / "results" / "rag" / "suggestions_A.jsonl", "review"),
            read_jsonl(ROOT / "results" / "rag" / "suggestions_A_autofiled.jsonl", "auto-filed")])),
        "test": (test_lab, pd.concat([
            read_jsonl(ROOT / "results" / "final_test" / "suggestions_A.jsonl", "review"),
            read_jsonl(ROOT / "results" / "scale_run" / "suggestions_A_autofiled.jsonl", "auto-filed")])),
    }

    pages = []
    for name, (lab, sug) in sets.items():
        df = lab.merge(sug, left_on="doc_id", right_on="id", how="left")
        missing = df["record_type"].isna().sum()
        if missing:
            raise SystemExit(f"{name}: {missing} labeled pages have no suggestion")
        df["set"] = name
        pages.append(df)
    df = pd.concat(pages, ignore_index=True)

    df = df.rename(columns={"record_type": "pred_type"})
    dept = lambda t: t.split(".")[0] if t else ""
    scored = lambda d: MERGE.get(d, d)
    df["true_department"] = df["type_id"].map(dept)
    df["pred_department"] = df["pred_type"].map(dept)
    df["also_department"] = df["also_fits"].map(dept)
    df["true_series"] = df["type_id"].map(series_of)
    df["pred_series"] = df["pred_type"].map(series_of)
    df["also_series"] = df["also_fits"].map(lambda t: series_of.get(t, ""))

    df["department_scored"] = df["pred_department"].map(scored) == df["true_department"].map(scored)
    df["department"] = df["pred_department"] == df["true_department"]
    df["series"] = df["pred_series"] == df["true_series"]
    df["record_type"] = df["pred_type"] == df["type_id"]
    df["record_type_top2"] = df["record_type"] | (df["second_record_type"] == df["type_id"])
    has_also = df["also_fits"] != ""
    df["department_or_also_fits"] = df["department"] | (has_also & (df["pred_department"] == df["also_department"]))
    df["series_or_also_fits"] = df["series"] | (has_also & (df["pred_series"] == df["also_series"]))
    df["record_type_or_also_fits"] = df["record_type"] | (has_also & (df["pred_type"] == df["also_fits"]))

    measures = ["department_scored", "department", "series", "record_type", "record_type_top2",
                "department_or_also_fits", "series_or_also_fits", "record_type_or_also_fits"]

    def table(group_cols):
        g = df.groupby(group_cols)
        t = g[measures].mean().round(3)
        t.insert(0, "pages", g.size())
        return t.reset_index()

    by_route = table(["set", "route"])
    all_rows = table(["set"])
    all_rows.insert(1, "route", "all")
    summary = pd.concat([by_route, all_rows]).sort_values(["set", "route"])
    summary.to_csv(OUT / "summary.csv", index=False)

    by_dept = table(["set", "true_department"])
    by_dept.to_csv(OUT / "by_department.csv", index=False)

    keep = ["set", "doc_id", "route", "certainty", "type_id", "also_fits", "pred_type", "second_record_type",
            "true_series", "pred_series"] + measures + ["reason"]
    df[[c for c in keep if c in df.columns]].to_csv(OUT / "per_page.csv", index=False)

    print(summary.to_string(index=False))
    print(f"\nSaved to {OUT}")


if __name__ == "__main__":
    main()
