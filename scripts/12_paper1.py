"""Paper 1 tables: reliability of extracted directions is language-dependent.

    python scripts/12_paper1.py

Pure re-analysis of existing CSVs. No model, no GPU.

Five tables:
  T1  fertility, floor and probe by language x model x pooling
  T2  the last-token advantage, against fertility
  T3  reliability against n (has it converged?)
  T4  the layer-selection trap: what a bare argmax would have picked
  T5  resampling unit: by-pair vs by-subject floors

THE TESTABILITY BOUND. A cosine between two independently estimated
directions is attenuated by the reliability of each. If arm A has split-half
reliability r_A and arm X has r_X, the highest cosine you could observe even
if the two directions were identical is approximately sqrt(r_A * r_X). So a
cell can only support a hypothesis test if that ceiling leaves room above the
floor's lower bound. This replaces the arbitrary 0.70 cut with a quantity
derived from the measurement itself.
"""
import argparse
import glob
import hashlib
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

pd.set_option("display.width", 240)
pd.set_option("display.max_rows", 400)

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "paper1"
OUT.mkdir(parents=True, exist_ok=True)

ap = argparse.ArgumentParser()
ap.add_argument("--min-clean", type=int, default=5,
                help="suppress pervasiveness below this many clean layers")
args = ap.parse_args()

SHORT = {"google/gemma-4-E2B": "Gemma4-E2B",
         "google/gemma-4-E4B": "Gemma4-E4B",
         "google/gemma-4-12B": "Gemma4-12B",
         "Jacaranda/AfroLlama_V1": "AfroLlama-8B"}
ORDER = ["Gemma4-E2B", "Gemma4-E4B", "Gemma4-12B", "AfroLlama-8B"]


def dedup(pattern):
    """Every match in the tree, once per distinct content."""
    seen, out = set(), []
    for f in glob.glob(str(ROOT / "**" / pattern), recursive=True):
        h = hashlib.md5(Path(f).read_bytes()).hexdigest()
        if h not in seen:
            seen.add(h)
            out.append(f)
    return sorted(out)


def split_run(tag):
    """'12B_mean' -> (model, pooling). Bare tags carry no pooling."""
    if not tag or tag == "untagged":
        return "?", "?"
    parts = tag.split("_")
    pool = parts[-1] if parts[-1] in ("last", "mean") else "?"
    model = "_".join(parts[:-1]) if pool != "?" else tag
    alias = {"12B": "Gemma4-12B", "E4B": "Gemma4-E4B", "E2B": "Gemma4-E2B",
             "afrollama": "AfroLlama-8B"}
    return alias.get(model, model), pool


# ============================================================ diagnostics
diag = []
for f in dedup("diagnostic_*.csv"):
    d = pd.read_csv(f)
    d["source"] = Path(f).name
    diag.append(d)

if not diag:
    sys.exit("no diagnostic_*.csv found")

dg = pd.concat(diag, ignore_index=True)
dg["model"] = dg["model"].map(lambda m: SHORT.get(m, str(m).split("/")[-1]))
dg = dg.drop_duplicates(subset=["lang", "model", "measure", "pool", "layer",
                                "value"], keep="first")

# drop any centring rows left over from earlier runs — a no-op for
# difference-of-means, so they duplicate the raw rows exactly
if "label" in dg.columns:
    dg = dg[~dg["label"].astype(str).str.contains("centred", na=False)]

fert = (dg[dg.measure == "fertility"][["lang", "model", "value"]]
        .drop_duplicates().rename(columns={"value": "fertility"}))

floors = dg[dg.measure == "floor"].copy()
probes = dg[dg.measure == "probe"].copy()

# ------------------------------------------------------------------ T1
vals = ["value", "lo"] + (["n_clean_layers"]
                          if "n_clean_layers" in floors else [])
t1 = floors.pivot_table(index=["lang", "model"], columns="pool",
                        values=vals).round(3)
rename = {"value": "floor", "lo": "floor_lo", "n_clean_layers": "n_clean"}
t1.columns = [f"{rename.get(a, a)}_{b}" for a, b in t1.columns]
p1 = probes.pivot_table(index=["lang", "model"], columns="pool",
                        values="value").round(3)
p1.columns = [f"probe_{c}" for c in p1.columns]
t1 = t1.join(p1).join(fert.set_index(["lang", "model"]))
t1 = t1.sort_values("fertility")
t1.to_csv(OUT / "T1_fertility_floor_probe.csv")

print("=" * 100)
print("T1  RELIABILITY AGAINST TOKENIZER FERTILITY")
print("    floor = split-half cosine at the preregistered layer (by subject)")
print("=" * 100)
print(t1.to_string())

# ------------------------------------------------------------------ T2
piv = floors.pivot_table(index=["lang", "model"], columns="pool",
                         values="value")
if {"last", "mean"} <= set(piv.columns):
    t2 = piv.assign(last_minus_mean=(piv["last"] - piv["mean"]).round(3))
    t2 = t2.join(fert.set_index(["lang", "model"])).sort_values("fertility")
    t2.to_csv(OUT / "T2_pooling_advantage.csv")
    print("\n" + "=" * 100)
    print("T2  THE LAST-TOKEN ADVANTAGE ERODES AS FERTILITY RISES")
    print("=" * 100)
    print(t2.round(3).to_string())
    ok = t2.dropna(subset=["last_minus_mean", "fertility"])
    if len(ok) > 2 and ok["last_minus_mean"].std() > 1e-9 \
            and ok["fertility"].std() > 1e-9:
        r = np.corrcoef(ok["fertility"], ok["last_minus_mean"])[0, 1]
        print(f"\n  correlation(fertility, last-minus-mean) = {r:+.3f} "
              f"over {len(ok)} cells")
        print("  n is small and cells share data; read as descriptive.")

# ------------------------------------------------------------------ T3
ns = dg[dg.measure == "n_scaling"]
if not ns.empty:
    t3 = ns.pivot_table(index=["lang", "model"], columns="n",
                        values="value").round(3)
    t3["still_rising"] = (t3.iloc[:, -1] > t3.iloc[:, -2]).map(
        {True: "yes", False: "no"})
    t3 = t3.join(fert.set_index(["lang", "model"])).sort_values("fertility")
    t3.to_csv(OUT / "T3_n_scaling.csv")
    print("\n" + "=" * 100)
    print("T3  RELIABILITY HAS NOT CONVERGED AT n=100")
    print("=" * 100)
    print(t3.to_string())

# ------------------------------------------------------------------ T4
if "bare_argmax_layer" in floors.columns:
    t4 = floors[["lang", "model", "pool", "layer", "value", "length_overlap",
                 "bare_argmax_layer", "bare_argmax_floor",
                 "bare_argmax_overlap", "n_clean_layers", "n_layers"]].copy()
    t4["rule_changes_layer"] = t4.layer != t4.bare_argmax_layer
    t4 = t4.sort_values("bare_argmax_overlap", ascending=False)
    t4.to_csv(OUT / "T4_layer_selection_trap.csv", index=False)
    print("\n" + "=" * 100)
    print("T4  THE LAYER-SELECTION TRAP")
    print("    a bare stability argmax prefers length-contaminated layers,")
    print("    because length is a strong and highly reproducible signal")
    print("=" * 100)
    print(t4.round(3).to_string(index=False))
    worst = t4.iloc[0]
    print(f"\n  worst case: {worst.lang} / {worst.model} / {worst['pool']} — "
          f"bare argmax picks L{int(worst.bare_argmax_layer)} at overlap "
          f"{worst.bare_argmax_overlap:.3f}")
    print(f"  the filtered rule picks L{int(worst.layer)} at overlap "
          f"{worst.length_overlap:.3f}")
else:
    print("\nT4 needs the patched 04_diagnose.py (bare_argmax_* columns)")

# ============================================================ per-layer
pl = []
for f in dedup("per_layer_*.csv"):
    m = re.match(r"per_layer_(\w\w\w)_(\w+?)(?:_(.+))?$", Path(f).stem)
    if not m:
        continue
    lang, _c, tag = m.groups()
    d = pd.read_csv(f)
    model, pool = split_run(tag)
    d = d.assign(lang=lang, model=model, pool=pool, source=Path(f).name)
    pl.append(d)

if pl:
    per = pd.concat(pl, ignore_index=True)
    # a file written before the by-subject change has no pairwise column
    per["has_both_floors"] = per["floor_lo_pairwise"].notna() \
        if "floor_lo_pairwise" in per else False

    # ------------------------------------------------------------- T5
    if per["has_both_floors"].any():
        both = per[per.has_both_floors]
        t5 = (both.groupby(["lang", "model", "pool"])
              .agg(layers=("layer", "count"),
                   floor_lo_subject=("floor_lo", "mean"),
                   floor_lo_pairwise=("floor_lo_pairwise", "mean"))
              .round(3))
        t5["inflation"] = (t5.floor_lo_pairwise - t5.floor_lo_subject).round(3)
        t5.to_csv(OUT / "T5_resampling_unit.csv")
        print("\n" + "=" * 100)
        print("T5  RESAMPLING UNIT: 4 pairs per subject")
        print("    pair-level resampling puts the same subject in both halves")
        print("=" * 100)
        print(t5.to_string())
        print(f"\n  mean inflation of floor_lo: {t5.inflation.mean():+.3f} "
              f"(range {t5.inflation.min():+.3f} to {t5.inflation.max():+.3f})")

    # ------------------------------------------- testability bound
    print("\n" + "=" * 100)
    print("T6  TESTABILITY: the attenuation ceiling sqrt(r_A * r_X)")
    print("    a cosine between two independently estimated directions cannot")
    print("    exceed the reliability of each; the ceiling is where a perfect")
    print("    match would land. Needs per-arm floors from 13_ceilings.py.")
    print("=" * 100)

    ceil_files = dedup("ceilings_*.csv")
    if not ceil_files:
        print("  no ceilings_*.csv found — run 13_ceilings.py per model/pooling")
    else:
        cl = pd.concat([pd.read_csv(f) for f in ceil_files], ignore_index=True)
        cl["model"] = cl["model"].map(lambda m: SHORT.get(m, str(m).split("/")[-1]))
        cl["arm"] = cl["comparison"].str[-1]
        if "pool" not in cl.columns:
            cl["pool"] = "?"

        # The floor is a cosine between two HALF samples; the cross-arm
        # cosines use full n. Classical test theory would correct that with
        # Spearman-Brown. Whether it transfers to cosines between
        # mean-difference vectors is an empirical question, so both are shown.
        def sb(r):
            r = np.clip(r, 0, None)
            return 2 * r / (1 + r)

        cl["ceiling_sb"] = np.sqrt(sb(cl.reliability_A) * sb(cl.reliability_X))
        cl["cos_over_ceiling_sb"] = cl.cosine / cl.ceiling_sb

        cols = ["lang", "model", "pool", "layer", "arm", "cosine",
                "reliability_A", "reliability_X", "ceiling",
                "cos_over_ceiling", "ceiling_sb", "cos_over_ceiling_sb"]
        t6 = cl[cols].sort_values(["lang", "model", "pool", "arm"])
        t6.to_csv(OUT / "T6_ceilings.csv", index=False)
        print(t6.round(3).to_string(index=False))

        # --- does the bound hold at all? -------------------------------
        print("\n  the ceiling is only a bound if the observed cosine stays "
              "under it:")
        for label, col in (("half-sample floor", "cos_over_ceiling"),
                           ("Spearman-Brown corrected", "cos_over_ceiling_sb")):
            n_bad = int((cl[col] > 1).sum())
            print(f"    {label:<26} exceeded in {n_bad}/{len(cl)} rows")
            for a, g in cl.groupby("arm"):
                print(f"      arm {a}: {int((g[col] > 1).sum())}/{len(g)}  "
                      f"median {g[col].median():.2f}")

        print("\n  Arms B and C behave as the attenuation model predicts once")
        print("  corrected. Arm D does not, and should not: D round-trips arm")
        print("  A's own sentences, so it is not an independent estimate of the")
        print("  same direction and the model's independence assumption fails")
        print("  by construction. Use the bound for B and C only.")

        # --- is D's lead a reliability artifact? -------------------------
        rel = cl.pivot_table(index=["lang", "model", "pool"], columns="arm",
                             values="reliability_X")
        if {"B", "C", "D"} <= set(rel.columns):
            d_top = ((rel.D > rel.B) & (rel.D > rel.C))
            print(f"\n  arm D is the most reliable arm in "
                  f"{int(d_top.sum())}/{len(rel)} cells; arm C beats it in "
                  f"{int((rel.C > rel.D).sum())}")

            changed = []
            for key, g in cl.groupby(["lang", "model", "pool"]):
                raw = g.loc[g.cosine.idxmax(), "arm"]
                nrm = g.loc[g.cos_over_ceiling.idxmax(), "arm"]
                if raw != nrm:
                    changed.append((key, raw, nrm))
            print(f"  closest arm changes under ceiling-normalisation in "
                  f"{len(changed)}/{len(rel)} cells")
            for key, raw, nrm in changed:
                print(f"    {key}: {raw} -> {nrm}")
            print("  If D leads on the normalised measure too, its closeness to")
            print("  native is not explained by D being easier to estimate.")

        # --- headroom, in bootstrap-spread units ------------------------
        pl_key = per[["lang", "model", "pool", "layer", "floor_mean",
                      "floor_lo"]].drop_duplicates() if pl else None
        if pl_key is not None:
            m = cl.merge(pl_key, on=["lang", "model", "pool", "layer"],
                         how="left")
            m["spread"] = m.floor_mean - m.floor_lo
            m["gap_in_spreads"] = (m.ceiling_sb - m.cosine) / m.spread
            gap = m[m.arm != "D"][["lang", "model", "pool", "arm", "cosine",
                                   "ceiling_sb", "spread", "gap_in_spreads"]]
            if not gap.empty:
                gap.to_csv(OUT / "T6b_headroom.csv", index=False)
                print("\n  headroom for arms B and C, in bootstrap-spread units")
                print("  (how far below a perfect match each arm sits; no")
                print("   threshold applied, so order them and judge)")
                print(gap.round(2).sort_values("gap_in_spreads")
                      .to_string(index=False))

    # --- clean-layer counts, reported not thresholded --------------------
    counts = []
    for (lang, model, pool), g in per.groupby(["lang", "model", "pool"]):
        n_clean = len(g[(g.length_overlap < 0.15)
                        & (g.arm == g.arm.iloc[0])])
        counts.append({"lang": lang, "model": model, "pool": pool,
                       "n_clean_layers": n_clean,
                       "n_layers": g.layer.nunique(),
                       "pervasiveness_reportable": n_clean >= args.min_clean})
    cdf = pd.DataFrame(counts).sort_values("n_clean_layers")
    cdf.to_csv(OUT / "T7_clean_layer_counts.csv", index=False)
    print("\n" + "=" * 100)
    print("T7  CLEAN-LAYER COUNTS")
    print(f"    cells with fewer than {args.min_clean} clean layers cannot")
    print("    support a pervasiveness fraction")
    print("=" * 100)
    print(cdf.to_string(index=False))

print(f"\nwrote tables to {OUT}")
