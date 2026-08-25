"""Does length explain the A-vs-C divergence? Tolerance sweep + regression.

    python scripts/07_length_control2.py --lang yor --concept sentiment

Three tests:

1. TOLERANCE SWEEP. Length-matched subsampling at several tolerances, each
   against a random subsample of the SAME size. If matching only kills the
   effect at one tolerance, it is a selection artifact, not length.

2. CONTINUOUS REGRESSION. Per layer, regress every activation on word count
   and keep the residuals. Uses the actual counts instead of a median split,
   and removes length from both arms with one model fitted on arm A only.

3. LENGTH OVERLAP PER LAYER. How much the concept direction already points
   along the length direction, across the whole stack — a property of the
   measurement, independent of provenance.
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
ap.add_argument("--lang", default="yor")
ap.add_argument("--concept", default="sentiment")
args = ap.parse_args()

cfg = Config.load()


def pair_lengths(arm):
    df = D.load_sheet(cfg.raw_dir / f"{args.lang}_{arm}.xlsx", args.lang, arm,
                      concept=args.concept)
    return ((df["positive"].str.split().str.len()
             + df["negative"].str.split().str.len()) / 2).to_numpy(float)


posA, negA, _, _ = E.load(cfg.act_dir / f"{args.lang}_A_{args.concept}.npz")
posC, negC, _, _ = E.load(cfg.act_dir / f"{args.lang}_C_{args.concept}.npz")
lenA, lenC = pair_lengths("A"), pair_lengths("C")

print(f"{args.lang} / {args.concept}")
print(f"arm A n={len(lenA)} mean {lenA.mean():.1f} | "
      f"arm C n={len(lenC)} mean {lenC.mean():.1f} "
      f"(gap {lenC.mean() - lenA.mean():+.1f})\n")


def verdict(pa, na, pc, nc, seed=None):
    """Returns (floor_mean, floor_lo, cosine, below) at the best layer."""
    f = V.split_half_floor(pa, na, n_splits=cfg.n_splits,
                           seed=cfg.seed if seed is None else seed)
    c = V.compare_arms((pa, na), (pc, nc))
    l = V.best_layer(f)
    return f["mean"][l], f["lo"][l], c[l], bool(c[l] < f["lo"][l]), l


rows = []
fm, lo, cos, below, L = verdict(posA, negA, posC, negC)
print(f"BASELINE            n=100  floor {fm:.3f} [lo {lo:.3f}]  "
      f"A-C {cos:.3f}  {'BELOW' if below else 'within noise'}  L{L}\n")
rows.append({"test": "baseline", "tolerance": None, "n": 100,
             "floor": fm, "floor_lo": lo, "cosine": cos, "below": below})

# --------------------------------------------------------- 1. sweep
print("1. TOLERANCE SWEEP  (matched vs random subsample of equal size)")
print(f"{'tol':>5} {'n':>4}  {'matched A-C':>12} {'lo':>7} {'':>7} "
      f"{'random A-C':>11} {'lo':>7}")
print("-" * 62)

for tol in (0.5, 1.0, 2.0, 3.0, 4.0, 6.0):
    used, keepA, keepC = set(), [], []
    for j in np.argsort(lenC):
        cand = [(abs(lenA[i] - lenC[j]), i) for i in range(len(lenA))
                if i not in used]
        if not cand:
            break
        d, i = min(cand)
        if d <= tol:
            used.add(i)
            keepA.append(i)
            keepC.append(j)
    if len(keepA) < 25:
        print(f"{tol:>5.1f} {len(keepA):>4}  (too few to test)")
        continue
    kA, kC = np.array(keepA), np.array(keepC)

    fm, lo, cos, below, _ = verdict(posA[kA], negA[kA], posC[kC], negC[kC])

    # random subsample of the same size, averaged over 5 draws
    rcos, rlo, rbelow = [], [], []
    for s in range(5):
        rng = np.random.default_rng(cfg.seed + s)
        i2 = rng.choice(len(lenA), size=len(kA), replace=False)
        j2 = rng.choice(len(lenC), size=len(kC), replace=False)
        _, l2, c2, b2, _ = verdict(posA[i2], negA[i2], posC[j2], negC[j2],
                                   seed=cfg.seed + s)
        rcos.append(c2)
        rlo.append(l2)
        rbelow.append(b2)

    print(f"{tol:>5.1f} {len(kA):>4}  {cos:>12.3f} {lo:>7.3f} "
          f"{'BELOW' if below else '  ---':>7} "
          f"{np.mean(rcos):>11.3f} {np.mean(rlo):>7.3f}  "
          f"({sum(rbelow)}/5 below)")
    rows.append({"test": "matched", "tolerance": tol, "n": len(kA),
                 "floor": fm, "floor_lo": lo, "cosine": cos, "below": below})
    rows.append({"test": "random_same_n", "tolerance": tol, "n": len(kA),
                 "floor": None, "floor_lo": float(np.mean(rlo)),
                 "cosine": float(np.mean(rcos)),
                 "below": bool(sum(rbelow) >= 3)})

# ------------------------------------------------ 2. continuous regression
print("\n2. CONTINUOUS LENGTH REGRESSION")


def regress_out(x, lengths, coef=None, mean_len=None):
    """Remove the linear effect of word count, per layer.

    Fits activation ~ a + b*length on arm A, then subtracts b*(length - mean)
    from both arms so the two are corrected with the SAME model.
    """
    n, n_layers, d = x.shape
    if coef is None:
        mean_len = lengths.mean()
        lc = (lengths - mean_len)[:, None]
        coef = np.zeros((n_layers, d), dtype=np.float64)
        denom = (lc ** 2).sum()
        for l in range(n_layers):
            coef[l] = (lc * (x[:, l, :] - x[:, l, :].mean(0))).sum(0) / denom
    lc = (lengths - mean_len)[:, None]
    out = x.astype(np.float64).copy()
    for l in range(x.shape[1]):
        out[:, l, :] -= lc * coef[l][None, :]
    return out.astype(np.float32), coef, mean_len


lenA_pair = np.repeat(lenA, 1)
rA, coef, mu = regress_out(np.concatenate([posA, negA]),
                           np.concatenate([lenA, lenA]))
pA2, nA2 = rA[:len(posA)], rA[len(posA):]
rC, _, _ = regress_out(np.concatenate([posC, negC]),
                       np.concatenate([lenC, lenC]), coef=coef, mean_len=mu)
pC2, nC2 = rC[:len(posC)], rC[len(posC):]

fm, lo, cos, below, L = verdict(pA2, nA2, pC2, nC2)
print(f"   after regressing out word count:  floor {fm:.3f} [lo {lo:.3f}]  "
      f"A-C {cos:.3f}  -> {'BELOW FLOOR' if below else 'within noise'}  L{L}")
rows.append({"test": "regressed", "tolerance": None, "n": 100,
             "floor": fm, "floor_lo": lo, "cosine": cos, "below": below})

# ------------------------------------------------ 3. overlap per layer
print("\n3. HOW MUCH IS THE CONCEPT DIRECTION ALREADY A LENGTH DIRECTION?")
med = np.median(lenA)
both = np.concatenate([posA, negA])
ml = np.concatenate([lenA, lenA])
ldir = both[ml > med].mean(0) - both[ml <= med].mean(0)
ldir /= np.clip(np.linalg.norm(ldir, axis=-1, keepdims=True), 1e-9, None)
overlap = np.abs(V.cosine(V.diff_in_means(posA, negA), ldir))
print(f"   |cos(concept, length)|  mean {overlap.mean():.3f}  "
      f"max {overlap.max():.3f} at L{int(np.argmax(overlap))}  "
      f"at best layer L{L}: {overlap[L]:.3f}")
for l in range(0, len(overlap), max(1, len(overlap) // 8)):
    print(f"      L{l:<3} {overlap[l]:.3f}")

out = cfg.results_dir / f"length_control2_{args.lang}_{args.concept}.csv"
pd.DataFrame(rows).to_csv(out, index=False)
print(f"\nwrote {out}")
