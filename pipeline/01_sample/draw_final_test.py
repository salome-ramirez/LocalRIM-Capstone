"""Draw the blind final-test pages from the scale batch.

The final test uses fresh pages from the scale batch (manifest positions 404-1403),
labeled without seeing any model output and scored once at the end. Pages are drawn at
random (seed 44) in shuffled order, with a reserve on top of the 150-label target so that
labeling in order still reaches 150 valid labels after exclusions. Excluded test pages are
kept: the routing check asks whether they go to review. Refuses to redraw if the output
exists.

Reads data/pm_sample/manifest.csv; writes results/final_test_pages.csv (order, id,
position, image).

Usage:
    python -m pipeline.01_sample.draw_final_test
"""

from __future__ import annotations


import pandas as pd

from pipeline.common import ROOT

MANIFEST = ROOT / "data" / "pm_sample" / "manifest.csv"
OUT = ROOT / "results" / "final_test_pages.csv"

FIRST, LAST = 404, 1403          # scale batch positions
TARGET = 150                     # valid test labels
RESERVE = 15                     # about 10% extra for exclusions
SEED = 44
NOT_RETRIEVED = {"qnkl0085", "hzgp0030", "xpxk0085"}   # never downloaded


def main() -> None:
    """Draw the final-test pages and save them."""
    if OUT.exists():
        raise SystemExit(f"{OUT.relative_to(ROOT)} already exists; the test set is drawn once.")
    manifest = pd.read_csv(MANIFEST, dtype=str)
    manifest["position"] = range(1, len(manifest) + 1)
    pool = manifest[(manifest["position"] >= FIRST) & (manifest["position"] <= LAST)
                    & ~manifest["id"].isin(NOT_RETRIEVED)]
    drawn = pool.sample(TARGET + RESERVE, random_state=SEED).reset_index(drop=True)
    drawn.insert(0, "order", range(1, len(drawn) + 1))
    OUT.parent.mkdir(exist_ok=True)
    drawn[["order", "id", "position", "image"]].to_csv(OUT, index=False)
    print(f"Drew {len(drawn)} pages ({TARGET} target + {RESERVE} reserve) from #{FIRST}-#{LAST}.")
    print(f"Saved {OUT.relative_to(ROOT)}. Label them in the labeler's 'Final test (blind)' mode.")


if __name__ == "__main__":
    main()
