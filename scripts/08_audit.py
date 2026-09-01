"""Every check, in one reproducible run. Writes CSVs, not just stdout.

    python scripts/08_audit.py --lang yor --concept sentiment --pool mean

Consolidates the checks that were previously done in ad-hoc notebook cells:

  per_layer.csv      A-vs-arm cosine, floor, and length overlap at EVERY layer
  matching_audit.csv what length-matching actually selects (the failed control)
  length_encoding.csv how strongly length is linearly encoded, per layer
  layer_choice.csv   candidate best-layer rules and what each one gives you
  summary.txt        human-readable log of the whole run

All outputs carry the pooling in their filename: activations are
pooling-specific, and a --pool mean run would otherwise overwrite the
--pool last results in place.

Layer-selection rule, fixed in advance:
  choose the layer with the highest split-half stability AMONG layers whose
  length overlap is below --max-overlap. The last layer usually has the
  highest raw stability but also the worst length contamination, so picking
  on stability alone selects the most confounded layer.
"""
import argparse
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from prov.config import Config  # noqa: E402
from prov import data as D  # noqa: E402
from prov import extract as E  # noqa: E402
from prov import vectors as V  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--lang", default="yor")
ap.add_argument("--concept", default="sentiment")
ap.add_argument("--max-overlap", type=float, default=0.15,
                help="length-overlap ceiling for the layer-selection rule")
ap.add_argument("--pool", default="mean", choices=["last", "mean"],
                help="which pooled activations to analyse")
ap.add_argument("--tag", default="")
args = ap.parse_args()

cfg = Config.load()
cfg.results_dir.mkdir(parents=True, exist_ok=True)
suffix = f"_{args.tag}_{args.pool}" if args.tag else f"_{args.pool}"
stem = f"{args.lang}_{args.concept}{suffix}"

log_lines = []


def log(s=""):
    print(s)
    log_lines.append(str(s))


log(f"audit run {datetime.now().isoformat(timespec='seconds')}")
log(f"model {cfg.model_id}  |  {args.lang} / {args.concept}  "
    f"|  pool {args.pool}")
log()


def pair_lengths(arm):
    df = D.load_sheet(cfg.raw_dir / f"{args.lang}_{arm}.xlsx", args.lang, arm,
                      concept=args.concept)
    return ((df["positive"].str.split().str.len()
             + df["negative"].str.split().str.len()) / 2).to_numpy(float)


# ---------------------------------------------------------------- load arms
arms, lengths = {}, {}
subjects = None
for arm in cfg.arms:
    p = E.act_path(cfg.act_dir, args.lang, arm, args.concept, args.pool)
    if not p.exists():
        continue
    # a pooling mismatch silently changes what is being measured
    E.check_meta(p, expect_pool=args.pool, expect_model=cfg.model_id)
    pos, neg, _, _, subj = E.load(p)
    arms[arm] = (pos, neg)
    if arm == "A":
        subjects = subj
    try:
        lengths[arm] = pair_lengths(arm)
    except Exception as exc:
        log(f"  (no lengths for arm {arm}: {exc})")

if "A" not in arms:
    sys.exit("no arm A activations — run 01_extract.py first")

log("arm    n   mean words")
for arm in arms:
    n = arms[arm][0].shape[0]
    ml = f"{lengths[arm].mean():.1f}" if arm in lengths else "?"
    log(f"  {arm}  {n:>3}   {ml:>6}")
log()

posA, negA = arms["A"]
lenA = lengths["A"]
n_layers = posA.shape[1]

# ------------------------------------------------------- length direction
med = np.median(lenA)
both = np.concatenate([posA, negA])
mlen = np.concatenate([lenA, lenA])
ldir = both[mlen > med].mean(0) - both[mlen <= med].mean(0)
ldir /= np.clip(np.linalg.norm(ldir, axis=-1, keepdims=True), 1e-9, None)

concept_dir = V.diff_in_means(posA, negA)
overlap = np.abs(V.cosine(concept_dir, ldir))

# linear encoding strength: correlation of the best single dim with length
lin_r = np.zeros(n_layers)
lc = (mlen - mlen.mean())
for l in range(n_layers):
    x = both[:, l, :]
    xc = x - x.mean(0)
    r = (lc[:, None] * xc).sum(0) / np.sqrt(
        (lc ** 2).sum() * (xc ** 2).sum(0) + 1e-12)
    lin_r[l] = np.abs(r).max()

pd.DataFrame({"layer": range(n_layers), "length_overlap": overlap,
              "max_abs_corr_dim_vs_length": lin_r}).to_csv(
    cfg.results_dir / f"length_encoding_{stem}.csv", index=False)

# ------------------------------------------------------------- per layer
floor_pair = V.split_half_floor(posA, negA, n_splits=cfg.n_splits,
                                seed=cfg.seed)
if subjects is not None:
    floor = V.split_half_floor_clustered(posA, negA, subjects=subjects,
                                         n_splits=cfg.n_splits, seed=cfg.seed)
else:
    floor = floor_pair
    log("no subject data — falling back to pair-level resampling")
rows = []
for arm in arms:
    if arm == "A":
        continue
    c = V.compare_arms((posA, negA), arms[arm])
    for l in range(n_layers):
        rows.append({
            "lang": args.lang, "concept": args.concept, "pool": args.pool,
            "model": cfg.model_id, "arm": arm, "layer": l,
            "cosine": c[l], "floor_mean": floor["mean"][l],
            "floor_lo": floor["lo"][l], "floor_hi": floor["hi"][l],
            "floor_lo_pairwise": floor_pair["lo"][l],
            "length_overlap": overlap[l],
            "below_floor": bool(c[l] < floor["lo"][l]),
        })

per_layer = pd.DataFrame(rows)
per_layer.to_csv(cfg.results_dir / f"per_layer_{stem}.csv", index=False)

# ----------------------------------------------------------- layer choice
clean = np.where(overlap < args.max_overlap)[0]
rules = {
    "max_stability_any_layer": int(np.argmax(floor["mean"])),
    f"max_stability_overlap_lt_{args.max_overlap}":
        int(clean[np.argmax(floor["mean"][clean])]) if len(clean) else -1,
    "min_length_overlap": int(np.argmin(overlap)),
}
choice_rows = []
for name, l in rules.items():
    if l < 0:
        continue
    entry = {"rule": name, "lang": args.lang, "model": cfg.model_id,
             "pool": args.pool, "layer": l, "floor": floor["mean"][l],
             "floor_lo": floor["lo"][l],
             "floor_lo_pairwise": floor_pair["lo"][l],
             "length_overlap": overlap[l]}
    for arm in arms:
        if arm == "A":
            continue
        sub = per_layer[(per_layer.arm == arm) & (per_layer.layer == l)]
        entry[f"cos_A_{arm}"] = float(sub["cosine"].iloc[0])
        entry[f"below_A_{arm}"] = bool(sub["below_floor"].iloc[0])
    choice_rows.append(entry)
choice = pd.DataFrame(choice_rows)
choice.to_csv(cfg.results_dir / f"layer_choice_{stem}.csv", index=False)

log("LAYER-SELECTION RULES")
log(choice.to_string(index=False))
log()

L = rules.get(f"max_stability_overlap_lt_{args.max_overlap}",
              rules["max_stability_any_layer"])
log(f"PREREGISTERED LAYER: L{L} "
    f"(max stability among layers with length overlap < {args.max_overlap})")
log(f"  floor {floor['mean'][L]:.3f} [{floor['lo'][L]:.3f}-{floor['hi'][L]:.3f}]"
    f"  length overlap {overlap[L]:.3f}")
log(f"  floor at L{L}: by-subject {floor['mean'][L]:.3f} "
    f"[{floor['lo'][L]:.3f}] | by-pair {floor_pair['mean'][L]:.3f} "
    f"[{floor_pair['lo'][L]:.3f}]  "
    f"(pair-level resampling inflates floor_lo by "
    f"{floor_pair['lo'][L] - floor['lo'][L]:+.3f})")
for arm in arms:
    if arm == "A":
        continue
    sub = per_layer[(per_layer.arm == arm)
                    & (per_layer.length_overlap < args.max_overlap)]
    if len(sub) < 3:
        log(f"  A vs {arm}: only {len(sub)} clean layers — spread not reported")
        continue
    log(f"  A vs {arm}: cosine across clean layers "
        f"min {sub.cosine.min():.3f} median {sub.cosine.median():.3f} "
        f"max {sub.cosine.max():.3f}  (n={len(sub)})")
log()

log("FRACTION OF LAYERS BELOW FLOOR")
for arm in arms:
    if arm == "A":
        continue
    sub = per_layer[per_layer.arm == arm]
    lowovl = sub[sub.length_overlap < args.max_overlap]
    if len(lowovl) < 5:
        log(f"  A vs {arm}: {sub['below_floor'].mean():.0%} of all layers; "
            f"only {len(lowovl)} clean layers, so the clean-layer fraction "
            f"is not reportable")
        continue
    log(f"  A vs {arm}: {sub['below_floor'].mean():.0%} of all layers, "
        f"{lowovl['below_floor'].mean():.0%} of low-length-overlap layers "
        f"(n={len(lowovl)})")
log()

# ------------------------------------------------------- matching audit
if "C" in arms and "C" in lengths:
    lenC = lengths["C"]
    m_rows = []
    for tol in (0.5, 1.0, 2.0, 3.0, 4.0, 6.0):
        used, kA, kC = set(), [], []
        for j in np.argsort(lenC):
            cand = [(abs(lenA[i] - lenC[j]), i)
                    for i in range(len(lenA)) if i not in used]
            if not cand:
                break
            d, i = min(cand)
            if d <= tol:
                used.add(i)
                kA.append(i)
                kC.append(j)
        if len(kA) < 25:
            continue
        kA, kC = np.array(kA), np.array(kC)

        # a length-matched subset can leave too few subjects to split
        if subjects is not None and len(np.unique(subjects[kA])) < 6:
            continue

        pctA = float((lenA < lenA[kA].mean()).mean())
        pctC = float((lenC < lenC[kC].mean()).mean())

        # same resampling scheme as the main table
        if subjects is not None:
            f2 = V.split_half_floor_clustered(
                posA[kA], negA[kA], subjects=subjects[kA],
                n_splits=cfg.n_splits, seed=cfg.seed)
        else:
            f2 = V.split_half_floor(posA[kA], negA[kA],
                                    n_splits=cfg.n_splits, seed=cfg.seed)

        c2 = V.compare_arms((posA[kA], negA[kA]),
                            (arms["C"][0][kC], arms["C"][1][kC]))

        # same layer rule as the main table: exclude length-contaminated layers
        clean_idx = np.where(overlap < args.max_overlap)[0]
        if len(clean_idx) == 0:
            continue
        l2 = int(clean_idx[np.argmax(f2["mean"][clean_idx])])
        
        m_rows.append({
            "tolerance": tol, "n": len(kA),
            "matched_len_A": lenA[kA].mean(), "matched_len_C": lenC[kC].mean(),
            "pctile_within_A": pctA, "pctile_within_C": pctC,
            "layer": l2, "floor": f2["mean"][l2], "floor_lo": f2["lo"][l2],
            "cosine": c2[l2], "below": bool(c2[l2] < f2["lo"][l2]),
        })
    match = pd.DataFrame(m_rows)
    match.to_csv(cfg.results_dir / f"matching_audit_{stem}.csv", index=False)
    log("LENGTH-MATCHING AUDIT (what the matched subsample actually selects)")
    log("  subsetting by length breaks the subject balance, so these floors")
    log("  are less well estimated than the full-sample one above")
    log(match.to_string(index=False))
    log()
    if len(match) and (match["pctile_within_A"].mean() > 0.65
                       and match["pctile_within_C"].mean() < 0.35):
        log("  NOTE: matching draws from the UPPER tail of arm A and the LOWER")
        log("  tail of arm C. It is selecting opposite ends of two distributions,")
        log("  not controlling length. Report as a failed control.")
        log()

(cfg.results_dir / f"summary_{stem}.txt").write_text("\n".join(log_lines))
print(f"wrote 4 csvs + summary_{stem}.txt to {cfg.results_dir}")
