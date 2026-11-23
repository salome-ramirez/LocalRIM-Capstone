"""Per-batch labeling metrics: one row per batch of 100.

Counted with localrim_metrics.py's definitions, the same ones the site uses. Run on its
own or through build_site_data.py, which also checks the two agree.

Writes results/batch_metrics.csv.

Usage:
    python -m pipeline.06_site_data.batch_metrics
"""
import pandas as pd

from pipeline.common import ROOT
import localrim_metrics as lm

OUT = ROOT / "results" / "batch_metrics.csv"


def write(out=None):
    """Count the metrics per batch, write them (unless out is False) and return the rows."""
    manifest = lm.load_manifest()
    saves = lm.all_saves()
    labels = lm.load_labels()
    skips = lm.load_skips(labels)
    labels["batch"] = lm.batch_of(labels["path"], manifest)
    skips["batch"] = lm.batch_of(skips["path"], manifest)
    labels["saved"] = pd.to_datetime(labels["saved_at"])

    rows = []
    for b in sorted(labels["batch"].dropna().unique()):
        g = labels[labels["batch"] == b].sort_values("saved")
        valid = g[g["outcome"] == "labeled"]
        head = lm.summary(g)
        gaps = g["saved"].diff().dt.total_seconds() / 60
        gaps = gaps[(gaps > 0) & (gaps < 15)]           # ignore breaks
        first = (b - 1) * lm.BATCH_SIZE + 1
        size = min(lm.BATCH_SIZE, len(manifest) - first + 1)
        n_skip = int((skips["batch"] == b).sum())
        resaved = saves[saves["path"].isin(g["path"])].groupby("path").size()
        row = {
            "batch": int(b),
            "status": "complete" if len(g) + n_skip >= size else "in progress",
            "documents": size,
            "reviewed": head["reviewed"],
            "skipped": n_skip,
            "valid": head["valid"],
            "excluded": head["excluded"],
            "no_fit": head["no_fit"],
            "unsure": head["unsure"],
            "unsure_share": head["unsure_share"],
            "secondary_share": round((valid["also_fits"] != "").mean(), 3) if len(valid) else None,
            "non_english": int(g["note"].str.contains("Non-English").sum()),
            "resaved_docs": int((resaved > 1).sum()),
            "median_min_between_saves": round(gaps.median(), 2) if len(gaps) else None,
            "first_saved": g["saved"].min().isoformat(timespec="minutes"),
            "last_saved": g["saved"].max().isoformat(timespec="minutes"),
        }
        for dept, n in valid["function_id"].value_counts().items():
            row[f"dept_{dept}"] = int(n)
        reasons = g.loc[g["outcome"] == "exclude", "note"].str.split(" | ", regex=False).str[0]
        for reason, n in reasons.value_counts().items():
            row[f"exclude_{reason}"] = int(n)
        rows.append(row)

    if out is not False:                       # False: check only, write nothing
        out = out or OUT
        out.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).fillna(0).to_csv(out, index=False)
    return rows


if __name__ == "__main__":
    rows = write()
    print(f"Wrote {OUT.relative_to(ROOT)}: {len(rows)} batches")
    for r in rows:
        print(f"  Batch {r['batch']} ({r['status']}): {r['valid']} valid, "
              f"{r['reviewed']} reviewed, {r['unsure_share']:.0%} unsure")
