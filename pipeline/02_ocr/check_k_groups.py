"""Elbow and silhouette check of the scan-quality groups; changes no group.

Runs k-means for k = 2 to 8 on the same standardised measures and seed as
profile_pages.py, recording inertia, silhouette and group sizes, plus the silhouette of
the five final groups. Also confirms that k = 4 reproduces the saved clusters.

Reads data/pm_sample/page_profile.csv; writes results/page_groups_k_check.csv.

Usage:
    python -m pipeline.02_ocr.check_k_groups
"""


import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import StandardScaler

from pipeline.common import ROOT

PROFILE = ROOT / "data" / "pm_sample" / "page_profile.csv"
OUT_CSV = ROOT / "results" / "page_groups_k_check.csv"

# Same measures and seed as profile_pages.py
FEATURES = ["brightness", "contrast", "background", "dark_share", "ink_share", "speckle", "sharpness"]
SEED = 44
K_VALUES = range(2, 9)


def main():
    """Run the k = 2 to 8 check and save it."""
    df = pd.read_csv(PROFILE, dtype={"id": str, "group": str})
    X = StandardScaler().fit_transform(df[FEATURES])

    rows = []
    for k in K_VALUES:
        km = KMeans(n_clusters=k, random_state=SEED, n_init=10).fit(X)
        sizes = sorted(np.bincount(km.labels_).tolist(), reverse=True)
        rows.append({
            "k": k,
            "inertia": round(km.inertia_, 1),
            "silhouette": round(silhouette_score(X, km.labels_), 3),
            "group_sizes": " / ".join(str(s) for s in sizes),
            "used": "yes" if k == 4 else "",
        })
        if k == 4:
            match = adjusted_rand_score(df["cluster"], km.labels_)
            print(f"k = 4 reproduces the saved clusters: agreement {match:.3f} (1.000 = identical)")

    # The five final groups (k = 4 plus the speckle split)
    final_sizes = df["group"].value_counts()
    rows.append({
        "k": "final",
        "inertia": "",
        "silhouette": round(silhouette_score(X, df["group"]), 3),
        "group_sizes": " / ".join(str(s) for s in sorted(final_sizes.tolist(), reverse=True)),
        "used": "final groups (k = 4 + speckle split)",
    })

    out = pd.DataFrame(rows)
    OUT_CSV.parent.mkdir(exist_ok=True)
    out.to_csv(OUT_CSV, index=False)
    print(out.to_string(index=False))
    print(f"\nSaved {OUT_CSV.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
