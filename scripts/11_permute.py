"""Permutation test: is each arm's distance from native explainable by chance?

    python scripts/11_permute.py --tag 12B

The floor test asks whether an arm sits further from native than sampling
noise. With floors below 0.70 in most cells, it cannot answer that. This asks
a different question that needs no floor:

    if provenance carried no signal, and each pair drew its translated version
    at random from the available arms, how often would the resulting arm sit
    as close to native as the real one does?

Arms B, C and D are row-matched to the same seed pairs, so a shuffled arm
holds the same content in the same order and differs only in which
provenance each row came from. That is the null.

Reported per arm:
  p_closer   fraction of shuffles at least as CLOSE to native as this arm.
             Low for D means D is closer than chance allows.
  p_further  fraction at least as FAR. Low for B or C means they are further
             than chance allows -- the provenance claim, without a floor.
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
ap.add_argument("--tag", default="", help="label for the output file")
ap.add_argument("--trials", type=int, default=1000)
ap.add_argument("--max-overlap", type=float, default=0.15)
args = ap.parse_args()

cfg = Config.load()
cfg.results_dir.mkdir(parents=True, exist_ok=True)
rows = []

for lang in cfg.languages:
    pA = cfg.act_dir / f"{lang}_A_sentiment.npz"
    if not pA.exists():
        continue
    posA, negA, _, _, subj = E.load(pA)

    arms = {}
    for a in ("B", "C", "D"):
        p = cfg.act_dir / f"{lang}_{a}_sentiment.npz"
        if p.exists():
            pos, neg = E.load(p)[:2]
            if len(pos) == len(posA):
                arms[a] = (pos, neg)
            else:
                print(f"{lang} arm {a}: {len(pos)} rows vs A's {len(posA)}, "
                      f"not row-matched — skipping")
    if len(arms) < 2:
        print(f"{lang}: need at least two arms to shuffle between")
        continue

    # ---- layer, by the same rule as the audit --------------------------
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
    floor = V.split_half_floor_clustered(posA, negA, subjects=subj,
                                         n_splits=cfg.n_splits, seed=cfg.seed)
    L = int(np.argmax(np.where(clean, floor["mean"], -np.inf)))

    print(f"\n=== {lang} @L{L} "
          f"(floor {floor['mean'][L]:.3f}, overlap {overlap[L]:.3f}, "
          f"{len(arms)} arms, {args.trials} shuffles) ===")

    # ---- null distribution ---------------------------------------------
    keys = list(arms)
    rng = np.random.default_rng(cfg.seed)
    n = len(posA)
    null = np.empty(args.trials)
    for t in range(args.trials):
        pick = rng.integers(0, len(keys), size=n)
        p_ = np.stack([arms[keys[pick[i]]][0][i] for i in range(n)])
        q_ = np.stack([arms[keys[pick[i]]][1][i] for i in range(n)])
        null[t] = V.cosine(concept_A, V.diff_in_means(p_, q_))[L]

    print(f"  null: mean {null.mean():.3f}  "
          f"2.5-97.5% [{np.percentile(null, 2.5):.3f}, "
          f"{np.percentile(null, 97.5):.3f}]")
    print(f"  {'arm':<4} {'cosine':>7} {'p_closer':>9} {'p_further':>10}  reading")

    for a in keys:
        obs = V.cosine(concept_A, V.diff_in_means(*arms[a]))[L]
        # +1 smoothing: a p of exactly 0 is not estimable from n trials
        p_close = (np.sum(null >= obs) + 1) / (args.trials + 1)
        p_far = (np.sum(null <= obs) + 1) / (args.trials + 1)
        if p_close < 0.05:
            reading = "closer to native than chance"
        elif p_far < 0.05:
            reading = "further from native than chance"
        else:
            reading = "indistinguishable from a random mixture"
        print(f"  {a:<4} {obs:>7.3f} {p_close:>9.3f} {p_far:>10.3f}  {reading}")
        rows.append({
            "lang": lang, "model": cfg.model_id, "layer": L, "arm": a,
            "cosine": obs, "floor": floor["mean"][L],
            "null_mean": null.mean(),
            "null_lo": np.percentile(null, 2.5),
            "null_hi": np.percentile(null, 97.5),
            "p_closer": p_close, "p_further": p_far,
            "n_arms_in_null": len(keys), "trials": args.trials,
        })

if not rows:
    sys.exit("nothing to test — need arm A plus two others, row-matched")

suffix = f"_{args.tag}" if args.tag else ""
out = cfg.results_dir / f"permutation{suffix}.csv"
pd.DataFrame(rows).to_csv(out, index=False)
print(f"\nwrote {out}")
print("\nNote: the null mixes the same arms being tested, so an arm sitting at")
print("the null mean is uninformative rather than absent of effect. With only")
print("two arms available the test is weak; three is the minimum useful.")
