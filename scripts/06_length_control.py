"""Is the A-vs-C divergence driven by sentence length rather than content?

    python scripts/06_length_control.py --lang yor --concept sentiment

Two independent controls:

1. MATCHED SUBSAMPLE. Greedily pair each C item with an unused A item of
   similar length, discard the rest, recompute. Costs sample size (which
   widens the floor) but makes no modelling assumption.

2. LENGTH-DIRECTION ABLATION. Build a "long vs short" direction from arm A
   alone, project it out of every activation, then recompute. Keeps all the
   data but assumes length is encoded roughly linearly.

If the divergence survives both, length is not the explanation.
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
ap.add_argument("--tolerance", type=float, default=2.0,
                help="max mean-word-count difference to call two pairs matched")
args = ap.parse_args()

cfg = Config.load()


def pair_lengths(lang, arm, concept):
    """Mean word count per pair, in the same row order as the activations."""
    df = D.load_sheet(cfg.raw_dir / f"{lang}_{arm}.xlsx", lang, arm,
                      concept=concept)
    return ((df["positive"].str.split().str.len()
             + df["negative"].str.split().str.len()) / 2).to_numpy()


def report(name, floor, c):
    layer = V.best_layer(floor)
    below = c[layer] < floor["lo"][layer]
    frac = (c < floor["lo"]).mean()
    print(f"{name}")
    print(f"   floor  {floor['mean'][layer]:.3f}  "
          f"[{floor['lo'][layer]:.3f}-{floor['hi'][layer]:.3f}]  L{layer}")
    print(f"   A vs C {c[layer]:.3f}  -> "
          f"{'BELOW FLOOR' if below else 'within noise'}  "
          f"({frac:.0%} of layers below)\n")
    return c[layer], floor["lo"][layer], below


posA, negA, _, _ = E.load(cfg.act_dir / f"{args.lang}_A_{args.concept}.npz")
posC, negC, _, _ = E.load(cfg.act_dir / f"{args.lang}_C_{args.concept}.npz")
lenA = pair_lengths(args.lang, "A", args.concept)
lenC = pair_lengths(args.lang, "C", args.concept)

print(f"{args.lang} / {args.concept}")
print(f"arm A: n={len(lenA)}  mean length {lenA.mean():.1f}")
print(f"arm C: n={len(lenC)}  mean length {lenC.mean():.1f}")
print(f"gap: {lenC.mean() - lenA.mean():+.1f} words\n")

rows = []

# ---------------------------------------------------------------- baseline
floor = V.split_half_floor(posA, negA, n_splits=cfg.n_splits, seed=cfg.seed)
c = V.compare_arms((posA, negA), (posC, negC))
cos, lo, below = report("BASELINE (no control)", floor, c)
rows.append({"control": "none", "n_A": len(lenA), "n_C": len(lenC),
             "cosine": cos, "floor_lo": lo, "below": below})

# ---------------------------------------------- 1. length-matched subsample
usedA, keepA, keepC = set(), [], []
order = np.argsort(lenC)
for j in order:
    best, best_d = None, np.inf
    for i in range(len(lenA)):
        if i in usedA:
            continue
        d = abs(lenA[i] - lenC[j])
        if d < best_d:
            best, best_d = i, d
    if best is not None and best_d <= args.tolerance:
        usedA.add(best)
        keepA.append(best)
        keepC.append(j)

keepA, keepC = np.array(keepA), np.array(keepC)
print(f"matched {len(keepA)} pairs within {args.tolerance} words")
print(f"   matched means: A {lenA[keepA].mean():.1f}  C {lenC[keepC].mean():.1f}"
      f"  (gap {lenC[keepC].mean() - lenA[keepA].mean():+.1f})\n")

if len(keepA) < 30:
    print("   too few matched pairs to be meaningful — widen --tolerance\n")
else:
    fm = V.split_half_floor(posA[keepA], negA[keepA], n_splits=cfg.n_splits,
                            seed=cfg.seed)
    cm = V.compare_arms((posA[keepA], negA[keepA]), (posC[keepC], negC[keepC]))
    cos, lo, below = report("1. LENGTH-MATCHED SUBSAMPLE", fm, cm)
    rows.append({"control": "matched_subsample", "n_A": len(keepA),
                 "n_C": len(keepC), "cosine": cos, "floor_lo": lo,
                 "below": below})

    # fair comparison: does a RANDOM subsample of the same size behave the same?
    rng = np.random.default_rng(cfg.seed)
    idx = rng.choice(len(lenA), size=len(keepA), replace=False)
    jdx = rng.choice(len(lenC), size=len(keepC), replace=False)
    fr = V.split_half_floor(posA[idx], negA[idx], n_splits=cfg.n_splits,
                            seed=cfg.seed)
    cr = V.compare_arms((posA[idx], negA[idx]), (posC[jdx], negC[jdx]))
    cos, lo, below = report(f"   (random subsample of n={len(idx)}, for scale)",
                            fr, cr)
    rows.append({"control": "random_subsample", "n_A": len(idx),
                 "n_C": len(jdx), "cosine": cos, "floor_lo": lo,
                 "below": below})

# ------------------------------------------- 2. length-direction ablation
def length_direction(pos, neg, lengths):
    """Direction separating long from short pairs, built from arm A only."""
    med = np.median(lengths)
    lng, sht = lengths > med, lengths <= med
    both = np.concatenate([pos, neg], axis=0)
    mask = np.concatenate([lng, lng]), np.concatenate([sht, sht])
    v = both[mask[0]].mean(0) - both[mask[1]].mean(0)
    return v / np.clip(np.linalg.norm(v, axis=-1, keepdims=True), 1e-9, None)


def ablate(x, direction):
    """Remove the component along `direction`, per layer."""
    proj = (x * direction[None, :, :]).sum(-1, keepdims=True)
    return x - proj * direction[None, :, :]


ldir = length_direction(posA, negA, lenA)
sim = V.cosine(V.diff_in_means(posA, negA), ldir)
print(f"2. LENGTH-DIRECTION ABLATION")
print(f"   |cos(concept direction, length direction)| at best layer: "
      f"{abs(sim[V.best_layer(floor)]):.3f}")
print("   (high = the concept direction already overlaps length)\n")

fa = V.split_half_floor(ablate(posA, ldir), ablate(negA, ldir),
                        n_splits=cfg.n_splits, seed=cfg.seed)
ca = V.compare_arms((ablate(posA, ldir), ablate(negA, ldir)),
                    (ablate(posC, ldir), ablate(negC, ldir)))
cos, lo, below = report("   after removing the length direction", fa, ca)
rows.append({"control": "length_ablated", "n_A": len(lenA), "n_C": len(lenC),
             "cosine": cos, "floor_lo": lo, "below": below})

out = cfg.results_dir / f"length_control_{args.lang}_{args.concept}.csv"
pd.DataFrame(rows).to_csv(out, index=False)
print(f"wrote {out}")
print("\nIf 'below' stays True across the controls, length is not the story.")
