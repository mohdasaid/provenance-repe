"""Separate current results from ones produced by earlier pipeline versions.

    python scripts/14_triage.py            # report only
    python scripts/14_triage.py --archive  # move stale files aside

Filenames cannot tell the two apart: the old `--tag 12B_mean` convention and
the new `--tag 12B --pool mean` both produce
`per_layer_yor_sentiment_12B_mean.csv`. The contents can.

A file is current if it carries the columns the patched scripts write:

  per_layer_*      pool, model            (added when 08_audit took --pool)
  layer_choice_*   pool, floor_lo_pairwise
  diagnostic_*     length_overlap, bare_argmax_layer
                                          (added when 04_diagnose gained the
                                           length-overlap filter)
  ceilings_*       pool
  arm_reliability_* pool

Anything without them predates the subject-level floor, the length-overlap
layer filter, or the pooling split, and mixing it with current output
produces tables that silently average two different methods.
"""
import argparse
import glob
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent

ap = argparse.ArgumentParser()
ap.add_argument("--archive", action="store_true",
                help="move stale files to results_archive/ (default: report)")
args = ap.parse_args()

REQUIRED = {
    "per_layer_": ["pool", "model"],
    "layer_choice_": ["pool"],
    "diagnostic_": ["length_overlap", "bare_argmax_layer"],
    "ceilings_": ["pool"],
    "arm_reliability_": ["pool"],
    "spearman_brown_": ["pool"],
    "permutation": ["pool"],
}

current, stale, unreadable = [], [], []

for prefix, cols in REQUIRED.items():
    for f in glob.glob(str(ROOT / "**" / f"{prefix}*.csv"), recursive=True):
        path = Path(f)
        if "results_archive" in path.parts:
            continue
        try:
            head = pd.read_csv(f, nrows=1)
        except Exception as exc:
            unreadable.append((path, str(exc)[:50]))
            continue
        missing = [c for c in cols if c not in head.columns]
        (stale if missing else current).append(
            (path, ",".join(missing) if missing else ""))

rel = lambda p: str(p.relative_to(ROOT))

print(f"current: {len(current)} files")
for p, _ in sorted(current):
    print(f"  {rel(p)}")

print(f"\nstale: {len(stale)} files (missing columns the patched scripts write)")
for p, missing in sorted(stale):
    print(f"  {rel(p):<70} missing: {missing}")

if unreadable:
    print(f"\nunreadable: {len(unreadable)}")
    for p, exc in unreadable:
        print(f"  {rel(p)}: {exc}")

# folders that exist only to hold old output
legacy_dirs = [d for d in ROOT.iterdir()
               if d.is_dir() and d.name.startswith("results_")
               and d.name != "results_archive"]
if legacy_dirs:
    print(f"\nlegacy result folders: "
          f"{', '.join(d.name for d in legacy_dirs)}")
    print("  these predate the current results/ layout and are the main")
    print("  source of duplicate rows in the collation scripts")

if not args.archive:
    print("\nreport only. rerun with --archive to move the stale files aside.")
    sys.exit(0)

arch = ROOT / "results_archive"
arch.mkdir(exist_ok=True)
moved = 0
for p, _ in stale:
    dest = arch / p.parent.name / p.name
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(p), str(dest))
    moved += 1
for d in legacy_dirs:
    dest = arch / d.name
    if dest.exists():
        shutil.rmtree(dest)
    shutil.move(str(d), str(dest))
    print(f"  moved folder {d.name}")

print(f"\nmoved {moved} files and {len(legacy_dirs)} folders to "
      f"{rel(arch)}")
print("nothing is deleted; the collation scripts skip results_archive/")
