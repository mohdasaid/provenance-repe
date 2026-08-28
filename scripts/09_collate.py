"""Collate every result CSV, wherever it landed, into three tables.

    python scripts/09_collate.py

Searches the whole tree, not just results/, because earlier runs were unzipped
into results_yor_E2B/, results_all_yor/ and similar. Deduplicates on content so
the same file appearing in three folders is counted once.

Writes results/TABLE1_provenance.csv, TABLE2_diagnostics.csv,
TABLE3_pervasiveness.csv and prints all three.
"""
import glob
import hashlib
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd  # noqa: E402

pd.set_option("display.width", 220)
pd.set_option("display.max_rows", 300)

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results"
OUT.mkdir(exist_ok=True)

# activation shape -> model, so a mislabelled config can be caught later
SHAPE_TO_MODEL = {(36, 1536): "gemma-4-E2B", (43, 2560): "gemma-4-E4B",
                  (49, 3840): "gemma-4-12B", (33, 4096): "AfroLlama_V1"}


def find(pattern):
    """Every matching file anywhere in the tree, deduplicated by content."""
    seen, out = {}, []
    for f in glob.glob(str(ROOT / "**" / pattern), recursive=True):
        h = hashlib.md5(Path(f).read_bytes()).hexdigest()
        if h not in seen:
            seen[h] = f
            out.append(f)
    return sorted(out)


def parse_name(path, prefix):
    """results/<prefix>_<lang>_<concept>_<tag>.csv -> (lang, tag)"""
    stem = Path(path).stem
    m = re.match(rf"{prefix}_(\w\w\w)_(\w+?)(?:_(.+))?$", stem)
    if not m:
        return "?", Path(path).parent.name
    lang, _concept, tag = m.groups()
    if not tag:                       # untagged file: fall back to folder name
        parent = Path(path).parent.name
        tag = parent.replace("results_", "") if parent != "results" else "12B"
    return lang, tag


def short(model):
    return str(model).split("/")[-1]


# ------------------------------------------------- TABLE 1: provenance
rows = []
for f in find("layer_choice_*.csv"):
    lang, tag = parse_name(f, "layer_choice")
    d = pd.read_csv(f)
    d = d[d["rule"].astype(str).str.startswith("max_stability_overlap")]
    if d.empty:
        continue
    d = d.copy()
    d.insert(0, "run", tag)
    d.insert(0, "lang", lang)
    d["source"] = str(Path(f).relative_to(ROOT))
    rows.append(d)

if rows:
    t1 = pd.concat(rows, ignore_index=True)
    cols = [c for c in ["lang", "run", "layer", "floor", "floor_lo",
                        "length_overlap", "cos_A_B", "below_A_B",
                        "cos_A_C", "below_A_C", "cos_A_D", "below_A_D"]
            if c in t1.columns]
    t1 = t1.sort_values(["lang", "run"])
    t1[cols + ["source"]].to_csv(OUT / "TABLE1_provenance.csv", index=False)
    print("=" * 100)
    print("TABLE 1  PROVENANCE at the preregistered layer")
    print("  below_A_C True  = translated pairs give a different direction")
    print("  below_A_D False = round-tripping native text does NOT")
    print("=" * 100)
    print(t1[cols].round(3).to_string(index=False))
    n = len(t1)
    print(f"\n  A-C below floor in {int(t1['below_A_C'].sum())}/{n} runs")
    if "below_A_D" in t1:
        print(f"  A-D below floor in {int(t1['below_A_D'].sum())}/{n} runs")
else:
    print("no layer_choice files found")

# ---------------------------------------------- TABLE 2: diagnostics
rows = []
for f in find("diagnostic_*.csv"):
    d = pd.read_csv(f)
    d["source"] = str(Path(f).relative_to(ROOT))
    rows.append(d)

print("\n" + "=" * 100)
print("TABLE 2  FLOORS, PROBES, FERTILITY")
print("=" * 100)
if rows:
    dg = pd.concat(rows, ignore_index=True)
    dg["model"] = dg["model"].map(short)
    fp = dg[dg.measure.isin(["floor", "probe"])].copy()
    if "label" in fp.columns:
        fp = fp[~fp["label"].astype(str).str.contains("centred", na=False)]
    piv = fp.pivot_table(index=["lang", "model"], columns=["measure", "pool"],
                         values="value").round(3)
    print(piv.to_string())

    fert = (dg[dg.measure == "fertility"][["lang", "model", "value"]]
            .drop_duplicates().round(3))
    print("\nfertility (tokens per word, both sides of each pair):")
    print(fert.to_string(index=False))

    ns = dg[dg.measure == "n_scaling"]
    if not ns.empty:
        print("\nfloor vs number of pairs:")
        print(ns.pivot_table(index=["lang", "model"], columns="n",
                             values="value").round(3).to_string())

    dg.to_csv(OUT / "TABLE2_diagnostics.csv", index=False)
    missing = [f"{l}/{m}" for l, m in
               [(l, m) for l in ("eng", "hau", "yor")
                for m in ("gemma-4-E2B", "gemma-4-E4B", "gemma-4-12B",
                          "AfroLlama_V1")]
               if not ((dg.lang == l) & (dg.model == m)).any()]
    if missing:
        print(f"\n  NOT RUN (or diagnostic CSV never written): "
              f"{', '.join(missing)}")
else:
    print("no diagnostic files found")

# ------------------------------------------- TABLE 3: pervasiveness
rows = []
for f in find("per_layer_*.csv"):
    lang, tag = parse_name(f, "per_layer")
    d = pd.read_csv(f)
    for arm, g in d.groupby("arm"):
        clean = g[g.length_overlap < 0.15]
        rows.append({"lang": lang, "run": tag, "arm": arm,
                     "n_layers": len(g),
                     "pct_all_below": g.below_floor.mean(),
                     "pct_clean_below": clean.below_floor.mean()
                     if len(clean) else float("nan"),
                     "n_clean": len(clean)})

print("\n" + "=" * 100)
print("TABLE 3  HOW PERVASIVE  (fraction of layers where A-vs-arm is below floor)")
print("  'clean' = layers whose length overlap is under 0.15")
print("=" * 100)
if rows:
    t3 = pd.DataFrame(rows).sort_values(["lang", "run", "arm"])
    t3.to_csv(OUT / "TABLE3_pervasiveness.csv", index=False)
    print(t3.round(2).to_string(index=False))
else:
    print("no per_layer files found")

print(f"\nwrote TABLE1/2/3 to {OUT}")
