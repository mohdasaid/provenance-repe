"""Per-arm reliability, attenuation ceilings, and ceiling-normalised cosines.

    python scripts/13_ceilings.py --tag 12B

Runs on saved activations. No forward passes.

WHY. A cosine between two independently estimated directions is bounded above
by the reliability of each. If arm A has split-half reliability r_A and arm X
has r_X, then even if the two underlying directions were identical you would
not observe a cosine much above

    ceiling = sqrt(r_A * r_X)

Comparing a raw cosine against arm A's floor alone ignores that arm X has its
own reliability. Two consequences:

1. A cell can only support a hypothesis test if the ceiling sits far enough
   above the floor's lower bound to leave detection room. Reported as a gap
   in units of the bootstrap spread, not as a threshold verdict:

       gap = (ceiling - cosine) / (floor_A - floor_lo_A)

2. Arms with higher self-reliability have more room to score highly. If arm D
   is more reliable than B or C -- plausible, since D round-trips arm A's own
   sentences and inherits their structure -- then part of D ranking first is
   mechanical. cos_normalised = cosine / ceiling removes that. If D still
   leads on the normalised measure, the ordering is not a reliability
   artifact.

SPEARMAN-BROWN. The floor is a cosine between two HALF samples while the
cross-arm cosines use full n, so the floor is biased low as a ceiling
estimate. Classical test theory would correct this as 2r/(1+r). Whether that
transfers to cosines between mean-difference vectors in high dimensions is an
empirical question, and the n-scaling data answers it directly: predict the
floor at n from the floor at n/2 and compare against what was observed. The
residual is reported rather than the correction applied.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from prov.config import Config  # noqa: E402
from prov import data as D  # noqa: E402
from prov import extract as E  # noqa: E402
from prov import vectors as V  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--tag", default="")
ap.add_argument("--max-overlap", type=float, default=0.15)
ap.add_argument("--pool", default="mean", choices=["last", "mean"],
                help="which pooled activations to analyse")
args = ap.parse_args()

cfg = Config.load()
cfg.results_dir.mkdir(parents=True, exist_ok=True)

arm_rows, cmp_rows, sb_rows = [], [], []

for lang in cfg.languages:
    pA = E.act_path(cfg.act_dir, lang, "A", "sentiment", args.pool)
    if not pA.exists():
        continue

    arms, subs = {}, {}
    for a in ("A", "B", "C", "D"):
        p = E.act_path(cfg.act_dir, lang, a, "sentiment", args.pool)
        if not p.exists():
            continue
        E.check_meta(p, expect_pool=args.pool, expect_model=cfg.model_id)
        pos, neg, _, _, subj = E.load(p)
        arms[a] = (pos, neg)
        subs[a] = subj

    if "A" not in arms or len(arms) < 2:
        continue

    posA, negA = arms["A"]

    # layer, by the audit's rule, from arm A
    df = D.load_sheet(cfg.raw_dir / f"{lang}_A.xlsx", lang, "A",
                      concept="sentiment")
    lens = ((df["positive"].str.split().str.len()
             + df["negative"].str.split().str.len()) / 2).to_numpy(float)
    both = np.concatenate([posA, negA])
    ml = np.concatenate([lens, lens])
    med = np.median(lens)
    ldir = both[ml > med].mean(0) - both[ml <= med].mean(0)
    ldir /= np.clip(np.linalg.norm(ldir, axis=-1, keepdims=True), 1e-9, None)
    concept_A = V.diff_in_means(posA, negA)
    overlap = np.abs(V.cosine(concept_A, ldir))
    clean = overlap < args.max_overlap

    # ---- per-arm floors -------------------------------------------------
    floors = {}
    for a, (pos, neg) in arms.items():
        s = subs[a]
        f = (V.split_half_floor_clustered(pos, neg, subjects=s,
                                          n_splits=cfg.n_splits, seed=cfg.seed)
             if s is not None else
             V.split_half_floor(pos, neg, n_splits=cfg.n_splits,
                                seed=cfg.seed))
        floors[a] = f

    L = int(np.argmax(np.where(clean, floors["A"]["mean"], -np.inf)))

    print(f"\n{'=' * 84}")
    print(f"{lang}  {cfg.model_id}  pool={args.pool}  L{L}  "
          f"({int(clean.sum())}/{len(clean)} clean layers)")
    print("=" * 84)
    print(f"  {'arm':<4} {'reliability':>12} {'95% lo':>8} {'95% hi':>8}")
    for a in sorted(floors):
        f = floors[a]
        print(f"  {a:<4} {f['mean'][L]:>12.3f} {f['lo'][L]:>8.3f} "
              f"{f['hi'][L]:>8.3f}")
        arm_rows.append({
            "lang": lang, "model": cfg.model_id, "layer": L, "arm": a,
            "reliability": f["mean"][L], "lo": f["lo"][L], "hi": f["hi"][L],
            "n_clean": int(clean.sum()), "n_layers": int(len(clean)),
        })

    rA = floors["A"]["mean"][L]
    spread = rA - floors["A"]["lo"][L]

    print(f"\n  {'cmp':<6} {'cosine':>8} {'ceiling':>8} {'norm':>7} "
          f"{'gap/spread':>11}   (ceiling = sqrt(r_A * r_X))")
    for a in sorted(arms):
        if a == "A":
            continue
        rX = floors[a]["mean"][L]
        ceiling = float(np.sqrt(max(rA, 0) * max(rX, 0)))
        cos = float(V.cosine(concept_A, V.diff_in_means(*arms[a]))[L])
        norm = cos / ceiling if ceiling > 1e-9 else np.nan
        gap = (ceiling - cos) / spread if spread > 1e-9 else np.nan
        print(f"  A-{a:<4} {cos:>8.3f} {ceiling:>8.3f} {norm:>7.3f} "
              f"{gap:>11.2f}")
        cmp_rows.append({
            "lang": lang, "model": cfg.model_id, "layer": L,
            "comparison": f"A_vs_{a}", "cosine": cos,
            "reliability_A": rA, "reliability_X": rX,
            "ceiling": ceiling, "cos_over_ceiling": norm,
            "gap_in_spreads": gap, "bootstrap_spread": spread,
        })

    # does normalising change the ordering?
    sub = [r for r in cmp_rows if r["lang"] == lang
           and r["model"] == cfg.model_id]
    if len(sub) >= 2:
        raw = sorted(sub, key=lambda r: -r["cosine"])[0]["comparison"]
        nrm = sorted(sub, key=lambda r: -r["cos_over_ceiling"])[0]["comparison"]
        note = "unchanged" if raw == nrm else f"CHANGES: {raw} -> {nrm}"
        print(f"\n  closest arm, raw vs ceiling-normalised: {note}")

    # ---- Spearman-Brown check ------------------------------------------
    # predict the floor at n from the floor at n/2 and compare with observed
    for n_half, n_full in ((25, 50), (50, len(posA))):
        fh = V.split_half_floor_clustered(
            posA[:n_half], negA[:n_half], subjects=subs["A"][:n_half],
            n_splits=cfg.n_splits, seed=cfg.seed)
        ff = V.split_half_floor_clustered(
            posA[:n_full], negA[:n_full], subjects=subs["A"][:n_full],
            n_splits=cfg.n_splits, seed=cfg.seed)
        r_h, r_f = fh["mean"][L], ff["mean"][L]
        pred = 2 * r_h / (1 + r_h) if r_h > -1 else np.nan
        sb_rows.append({
            "lang": lang, "model": cfg.model_id, "layer": L,
            "n_half": n_half, "n_full": n_full,
            "r_half": r_h, "sb_predicted": pred, "r_observed": r_f,
            "residual": r_f - pred,
        })

    print("\n  Spearman-Brown check (predicting the floor at n from n/2)")
    for r in sb_rows[-2:]:
        print(f"    n {r['n_half']:>3}->{r['n_full']:<4} r_half "
              f"{r['r_half']:.3f}  predicted {r['sb_predicted']:.3f}  "
              f"observed {r['r_observed']:.3f}  "
              f"residual {r['residual']:+.3f}")

if not arm_rows:
    sys.exit("no activations found — run 01_extract.py first")

suffix = f"_{args.tag}_{args.pool}" if args.tag else f"_{args.pool}"
for frame, name in ((arm_rows, "arm_reliability"), (cmp_rows, "ceilings"),
                    (sb_rows, "spearman_brown")):
    d = pd.DataFrame(frame)
    d["pool"] = args.pool
    d.to_csv(cfg.results_dir / f"{name}{suffix}.csv", index=False)

sb = pd.DataFrame(sb_rows)
print(f"\n{'=' * 84}")
print("SPEARMAN-BROWN, all cells")
print(f"  mean residual {sb.residual.mean():+.3f}  "
      f"(negative = the correction overpredicts)")
print(f"  overpredicts in {int((sb.residual < 0).sum())} of {len(sb)} cells")
print("  With this few points and two model families, read as descriptive.")
print(f"\nwrote 3 csvs to {cfg.results_dir}")
