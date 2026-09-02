"""Re-analysis of existing results. No model, no re-extraction.

    python scripts/10_reanalyse.py

Three changes to how the same numbers are reported:

1. THREE-WAY VERDICTS. "cosine < floor_lo" collapses two different states:
   a cell where the arms genuinely agree, and a cell where the floor is so
   low that nothing could have been detected. Cells whose floor falls below
   --min-floor are labelled "untestable" rather than counted as nulls.

2. BOTH FLOORS SIDE BY SIDE. Pair-level resampling puts the same subject in
   both halves (4 pairs per subject), so it overstates stability. Reported at
   every cell, not just the preregistered layer.

3. RANK AGREEMENT. A binary flag discards the ordering. The prediction is
   that arm D (native text round-tripped) sits closest to arm A, and the
   translated arms further away. Rank agreement measures whether that
   ordering holds even where no arm crosses the threshold.
"""
import argparse
import glob
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

pd.set_option("display.width", 220)
pd.set_option("display.max_rows", 400)

ROOT = Path(__file__).resolve().parent.parent

ap = argparse.ArgumentParser()
ap.add_argument("--min-floor", type=float, default=0.70,
                help="floors below this are untestable, not null")
ap.add_argument("--max-overlap", type=float, default=0.15)
args = ap.parse_args()

OUT = ROOT / "results"
OUT.mkdir(exist_ok=True)


def parse_name(path):
    """per_layer_<lang>_<concept>_<tag>.csv -> (lang, tag)"""
    m = re.match(r"per_layer_(\w\w\w)_(\w+?)(?:_(.+))?$", Path(path).stem)
    if not m:
        return "?", "?"
    lang, _concept, tag = m.groups()
    return lang, tag or "untagged"


def verdict(cosine, floor_mean, floor_lo, min_floor):
    """Three states, not two."""
    if floor_mean < min_floor:
        return "untestable"
    return "below floor" if cosine < floor_lo else "above floor"


# ------------------------------------------------------------ gather cells
files = [f for f in sorted(glob.glob(str(ROOT / "**" / "per_layer_*.csv"),
                                     recursive=True))
         if "results_archive" not in Path(f).parts]

# files from earlier pipeline versions lack the columns the patched audit
# writes; mixing them in averages two different methods
_kept = []
for f in files:
    try:
        head = pd.read_csv(f, nrows=1)
    except Exception:
        continue
    if {"pool", "model"} <= set(head.columns):
        _kept.append(f)
    else:
        print(f"  skipping stale {Path(f).name} (no pool/model column)")
files = _kept
if not files:
    sys.exit("no per_layer_*.csv found — run 08_audit.py first")

cells, layer_rows = [], []
for f in files:
    lang, tag = parse_name(f)
    d = pd.read_csv(f)
    if "floor_lo_pairwise" not in d.columns:
        d["floor_lo_pairwise"] = np.nan

    clean = d[d.length_overlap < args.max_overlap]
    if clean.empty:
        continue

    # preregistered layer: most stable among clean layers
    L = int(clean.loc[clean.floor_mean.idxmax(), "layer"])

    for arm, g in d.groupby("arm"):
        row = g[g.layer == L]
        if row.empty:
            continue
        row = row.iloc[0]
        gc = g[g.length_overlap < args.max_overlap]
        cells.append({
            "lang": lang, "run": tag, "arm": arm, "layer": L,
            "floor_subject": row.floor_mean,
            "floor_lo_subject": row.floor_lo,
            "floor_lo_pairwise": row.floor_lo_pairwise,
            "floor_inflation": (row.floor_lo_pairwise - row.floor_lo
                                if pd.notna(row.floor_lo_pairwise) else np.nan),
            "cosine": row.cosine,
            "cos_median_clean": gc.cosine.median(),
            "verdict": verdict(row.cosine, row.floor_mean, row.floor_lo,
                               args.min_floor),
            "verdict_pairwise": verdict(row.cosine, row.floor_mean,
                                        row.floor_lo_pairwise, args.min_floor)
            if pd.notna(row.floor_lo_pairwise) else "n/a",
            "pct_clean_below": (gc.cosine < gc.floor_lo).mean(),
            "n_clean": len(gc),
        })
    layer_rows.append(d.assign(lang=lang, run=tag))

cells = pd.DataFrame(cells)
cells.to_csv(OUT / "REANALYSIS_cells.csv", index=False)

print("=" * 110)
print("1. THREE-WAY VERDICTS at the preregistered layer")
print(f"   floors below {args.min_floor} are 'untestable', not null")
print("=" * 110)
show = ["lang", "run", "arm", "layer", "floor_subject", "floor_lo_subject",
        "cosine", "verdict", "pct_clean_below", "n_clean"]
print(cells[show].round(3).to_string(index=False))

print("\ncount by verdict:")
print(cells.groupby(["arm", "verdict"]).size().unstack(fill_value=0).to_string())

# ------------------------------------------------- 2. floor inflation
print("\n" + "=" * 110)
print("2. RESAMPLING UNIT: by-pair vs by-subject")
print("   4 pairs per subject, so pair-level resampling puts the same subject")
print("   in both halves and overstates stability.")
print("=" * 110)
inf = cells.dropna(subset=["floor_inflation"])
if inf.empty:
    print("  no pairwise floors recorded — rerun 08_audit.py to populate them")
else:
    per_run = (inf.groupby(["lang", "run"])
               .agg(floor_lo_subject=("floor_lo_subject", "first"),
                    floor_lo_pairwise=("floor_lo_pairwise", "first"),
                    inflation=("floor_inflation", "first")).round(3))
    print(per_run.to_string())
    print(f"\n  mean inflation of floor_lo: "
          f"{inf['floor_inflation'].mean():+.3f}")
    flipped = inf[inf.verdict != inf.verdict_pairwise]
    print(f"  cells whose verdict changes with resampling unit: "
          f"{len(flipped)} of {len(inf)}")
    if len(flipped):
        print(flipped[["lang", "run", "arm", "cosine", "verdict",
                       "verdict_pairwise"]].round(3).to_string(index=False))

# ------------------------------------------------- 3. rank agreement
print("\n" + "=" * 110)
print("3. RANK AGREEMENT  (does the ordering hold where thresholds do not?)")
print("   prediction: D closest to A, translated arms further")
print("=" * 110)

rank_rows = []
for (lang, run), g in cells.groupby(["lang", "run"]):
    arms = g.set_index("arm")["cosine"].to_dict()
    if "D" not in arms or len(arms) < 2:
        continue
    others = {a: c for a, c in arms.items() if a != "D"}
    d_highest = all(arms["D"] > c for c in others.values())
    margin = arms["D"] - max(others.values())
    rank_rows.append({
        "lang": lang, "run": run,
        **{f"cos_{a}": round(c, 3) for a, c in sorted(arms.items())},
        "D_ranks_first": d_highest,
        "margin_over_next": round(margin, 3),
        "any_below_floor": bool((g.verdict == "below floor").any()),
        "testable": bool((g.verdict != "untestable").any()),
    })

if rank_rows:
    rk = pd.DataFrame(rank_rows)
    rk.to_csv(OUT / "REANALYSIS_rank.csv", index=False)
    print(rk.to_string(index=False))
    n = len(rk)
    print(f"\n  D ranks first in {int(rk.D_ranks_first.sum())}/{n} cells")
    print(f"  mean margin over the nearest translated arm: "
          f"{rk.margin_over_next.mean():+.3f}")
    silent = rk[(rk.D_ranks_first) & (~rk.any_below_floor)]
    print(f"  cells where the ordering holds but NO arm crosses the "
          f"threshold: {len(silent)}")
    if len(silent):
        print("    (these are the cells a binary flag would have discarded)")
        print(silent[["lang", "run", "margin_over_next"]].to_string(index=False))
else:
    print("  need arm D plus at least one other arm per cell")

print(f"\nwrote REANALYSIS_cells.csv and REANALYSIS_rank.csv to {OUT}")
