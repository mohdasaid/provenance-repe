"""Permutation tests on arm provenance, with the confounds handled.

    python scripts/11_permute.py --tag 12B --trials 10000

WHAT ARM D ACTUALLY IS. Arm D round-trips arm A's OWN realised sentences
through English and back. It therefore shares item-level wording, topic
choice and idiom with arm A that arms B and C never had. So "D is close to A"
partly confirms the construction rather than testing provenance. D is a
control for translation MECHANICS with content held constant, not a control
for translation in general. Read every D result with that in mind.

The load-bearing comparison is B against C, and B/C against a null that does
not contain them.

THREE TESTS, in increasing cleanliness:

  full      null mixes every available arm. The arm under test is inside its
            own null, so a high arm mechanically pushes the others below it.
            Retained for continuity; not evidence on its own.

  loo_X     leave-one-out: the null is built WITHOUT arm X, then X is tested
            against it. Removes that contamination. This is the one to report.

  bc_only   null from B and C only, testing each of them. Uses no
            native-derived arm, so arm D's construction cannot influence it.

SHUFFLING UNIT. Pairs cluster in subjects (4 per subject). Shuffling
individual items breaks that clustering and narrows the null -- the same
non-independence that inflated pair-level floors. Default is subject-level;
--unit item reproduces the old behaviour for comparison.
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
ap.add_argument("--trials", type=int, default=10000)
ap.add_argument("--max-overlap", type=float, default=0.15)
ap.add_argument("--unit", default="subject", choices=["subject", "item"],
                help="shuffle whole subjects (default) or individual pairs")
ap.add_argument("--pool", default="mean", choices=["last", "mean"],
                help="which pooled activations to analyse")
ap.add_argument("--layer-window", type=int, default=1,
                help="also test L+-k around the chosen layer")
args = ap.parse_args()

cfg = Config.load()
cfg.results_dir.mkdir(parents=True, exist_ok=True)
rows = []


def build_null(concept_A, sources, subjects, layer, trials, unit, seed):
    """Distribution of cosine(A, mixed_arm) under random provenance.

    subject unit: every pair of a subject takes the same source arm, so the
    clustering present in the real data is preserved in the null.
    """
    rng = np.random.default_rng(seed)
    keys = list(sources)
    n = sources[keys[0]][0].shape[0]
    subs = np.unique(subjects)
    idx_by_sub = {s: np.where(subjects == s)[0] for s in subs}

    null = np.empty(trials)
    for t in range(trials):
        pick = np.empty(n, dtype=int)
        if unit == "subject":
            for s in subs:
                pick[idx_by_sub[s]] = rng.integers(0, len(keys))
        else:
            pick[:] = rng.integers(0, len(keys), size=n)
        p_ = np.stack([sources[keys[pick[i]]][0][i] for i in range(n)])
        q_ = np.stack([sources[keys[pick[i]]][1][i] for i in range(n)])
        null[t] = V.cosine(concept_A, V.diff_in_means(p_, q_))[layer]
    return null


def pvals(null, obs, trials):
    """Two-sided, +1 smoothed: p=0 is not estimable from finite trials."""
    return ((np.sum(null >= obs) + 1) / (trials + 1),
            (np.sum(null <= obs) + 1) / (trials + 1))


for lang in cfg.languages:
    pA = E.act_path(cfg.act_dir, lang, "A", "sentiment", args.pool)
    if not pA.exists():
        continue
    E.check_meta(pA, expect_pool=args.pool, expect_model=cfg.model_id)
    posA, negA, _, _, subj = E.load(pA)

    arms = {}
    for a in ("B", "C", "D"):
        p = E.act_path(cfg.act_dir, lang, a, "sentiment", args.pool)
        if p.exists():
            E.check_meta(p, expect_pool=args.pool, expect_model=cfg.model_id)
            pos, neg = E.load(p)[:2]
            if len(pos) == len(posA):
                arms[a] = (pos, neg)
            else:
                print(f"{lang} arm {a}: not row-matched, skipping")
    if len(arms) < 2:
        continue

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
    L0 = int(np.argmax(np.where(clean, floor["mean"], -np.inf)))
    layers = [l for l in range(L0 - args.layer_window,
                               L0 + args.layer_window + 1)
              if 0 <= l < posA.shape[1]]

    print(f"\n{'=' * 78}")
    print(f"{lang}  model {cfg.model_id}  arms {sorted(arms)}")
    print(f"  chosen layer L{L0} (floor {floor['mean'][L0]:.3f}, "
          f"overlap {overlap[L0]:.3f}); also testing {layers}")
    print(f"  pooling {args.pool}")
    print(f"  shuffling by {args.unit}, {args.trials} trials, "
          f"{len(np.unique(subj))} subjects")
    print("=" * 78)

    for L in layers:
        mark = "  <- chosen" if L == L0 else ""
        print(f"\n L{L}  floor {floor['mean'][L]:.3f}  "
              f"overlap {overlap[L]:.3f}{mark}")
        obs = {a: V.cosine(concept_A, V.diff_in_means(*arms[a]))[L]
               for a in arms}
        print("  observed: " + "  ".join(f"{a}={obs[a]:.3f}"
                                         for a in sorted(obs)))

        tests = [("full", list(arms), list(arms))]
        for a in arms:
            others = [k for k in arms if k != a]
            if len(others) >= 2:
                tests.append((f"loo_{a}", others, [a]))
        if "B" in arms and "C" in arms and len(arms) > 2:
            tests.append(("bc_only", ["B", "C"], ["B", "C"]))

        for name, src_keys, targets in tests:
            sources = {k: arms[k] for k in src_keys}
            null = build_null(concept_A, sources, subj, L, args.trials,
                              args.unit, cfg.seed)
            lo, hi = np.percentile(null, [2.5, 97.5])

            for a in targets:
                pc, pf = pvals(null, obs[a], args.trials)
                verdict = ("closer than chance" if pc < 0.05 else
                           "further than chance" if pf < 0.05 else
                           "indistinguishable")
                flag = " [inside own null]" if a in src_keys else ""
                if a == "D":
                    flag += " [D shares A's wording]"
                print(f"   {name:<9} null={'+'.join(sorted(src_keys)):<5} "
                      f"arm {a}  null {null.mean():.3f} [{lo:.3f},{hi:.3f}]  "
                      f"p_closer {pc:.4f}  p_further {pf:.4f}  "
                      f"{verdict}{flag}")
                rows.append({
                    "lang": lang, "model": cfg.model_id, "layer": L,
                    "chosen_layer": L0, "test": name,
                    "null_sources": "+".join(sorted(src_keys)),
                    "arm_in_own_null": a in src_keys,
                    "arm": a, "cosine": obs[a], "floor": floor["mean"][L],
                    "null_mean": null.mean(), "null_lo": lo, "null_hi": hi,
                    "p_closer": pc, "p_further": pf,
                    "margin_to_null_hi": obs[a] - hi,
                    "pool": args.pool, "unit": args.unit,
                    "trials": args.trials,
                })

if not rows:
    sys.exit("nothing to test — need arm A plus two row-matched arms")

suffix = f"_{args.tag}" if args.tag else ""
out = cfg.results_dir / f"permutation{suffix}_{args.pool}_{args.unit}.csv"
pd.DataFrame(rows).to_csv(out, index=False)
print(f"\nwrote {out}")
print("\nReport the loo_ and bc_only rows. The full rows keep each arm inside")
print("its own null, so they overstate how extreme the others look.")
print("margin_to_null_hi shows how far above the null's upper bound an arm")
print("sits — a p of 0.005 with a margin of 0.01 is on the line.")
