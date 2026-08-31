"""The whole result, from saved activations. Per language, per concept.

    python scripts/02_analyze.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd  # noqa: E402

from prov.config import Config  # noqa: E402
from prov import data as D  # noqa: E402
from prov import extract as E  # noqa: E402
from prov import vectors as V  # noqa: E402

cfg = Config.load()
cfg.results_dir.mkdir(parents=True, exist_ok=True)

import argparse
ap = argparse.ArgumentParser()
ap.add_argument("--tag", default="", help="suffix for output files, e.g. mean")
args = ap.parse_args()
suffix = f"_{args.tag}" if args.tag else ""

floor_rows, cos_rows, probe_rows = [], [], []
found_any = False

for lang in cfg.languages:
    for concept in D.CONCEPT_SHEETS:
        arms = {}
        subjects = None
        for arm in cfg.arms:
            p = cfg.act_dir / f"{lang}_{arm}_{concept}.npz"
            if p.exists():
                pos, neg, _, _, subj = E.load(p)
                arms[arm] = (pos, neg)
                if arm == "A":
                    subjects = subj

        if "A" not in arms:
            continue
        found_any = True

        # --- noise floor, arm A against itself --------------------------
        if subjects is not None:
            floor = V.split_half_floor_clustered(
                *arms["A"], subjects=subjects,
                n_splits=cfg.n_splits, seed=cfg.seed)
        else:
            floor = V.split_half_floor(*arms["A"], n_splits=cfg.n_splits,
                                       seed=cfg.seed)
        L = len(floor["mean"])
        for l in range(L):
            floor_rows.append({
                "lang": lang, "concept": concept, "layer": l,
                "floor_mean": floor["mean"][l],
                "floor_lo": floor["lo"][l], "floor_hi": floor["hi"][l],
            })

        layer = V.best_layer(floor)
        n_pairs = arms["A"][0].shape[0]
        print(f"\n=== {lang} / {concept}  (n={n_pairs} pairs) ===")
        print(f"most stable layer: {layer}  "
              f"split-half cosine {floor['mean'][layer]:.3f}  "
              f"95% range {floor['lo'][layer]:.3f}-{floor['hi'][layer]:.3f}")

        if floor["mean"][layer] < 0.70:
            print("  WARNING: same-provenance halves disagree badly. Either n is")
            print("  too small or this concept is not linearly encoded here.")
            print("  No cross-arm comparison will mean anything until this improves.")
        elif floor["mean"][layer] < 0.80:
            print("  CAUTION: floor is marginal. Treat comparisons as provisional.")

        others = [a for a in arms if a != "A"]
        if not others:
            print("  (no other arms yet — floor only)")
            continue

        for arm in others:
            c = V.compare_arms(arms["A"], arms[arm])
            below = V.verdict(c, floor)
            for l in range(L):
                cos_rows.append({
                    "lang": lang, "concept": concept,
                    "comparison": f"A_vs_{arm}", "layer": l,
                    "cosine": c[l], "floor_lo": floor["lo"][l],
                    "below_floor": bool(below[l]),
                })
            mark = "BELOW FLOOR" if below[layer] else "within noise"
            frac = below.mean()
            print(f"  A vs {arm} @layer{layer}: cos={c[layer]:.3f}  "
                  f"floor_lo={floor['lo'][layer]:.3f}  -> {mark}  "
                  f"({frac:.0%} of layers below)")

            acc_self = V.probe_transfer(arms["A"], arms["A"], layer, cfg.seed)
            acc_cross = V.probe_transfer(arms["A"], arms[arm], layer, cfg.seed)
            probe_rows.append({
                "lang": lang, "concept": concept, "train": "A", "test": arm,
                "layer": layer, "acc_within_A": acc_self,
                "acc_cross": acc_cross, "drop": acc_self - acc_cross,
            })
            note = "  (saturated - uninformative)" if acc_self > 0.99 else ""
            print(f"       probe A->A {acc_self:.3f} | A->{arm} {acc_cross:.3f}"
                  f"{note}")

if not found_any:
    sys.exit("no arm A activations found — run 01_extract.py first")

pd.DataFrame(floor_rows).to_csv(cfg.results_dir / f"noise_floor{suffix}.csv", index=False)
if cos_rows:
    pd.DataFrame(cos_rows).to_csv(cfg.results_dir / f"arm_cosines{suffix}.csv", index=False)
    pd.DataFrame(probe_rows).to_csv(cfg.results_dir / f"probe_transfer{suffix}.csv",
                                    index=False)
print(f"\nwrote results to {cfg.results_dir}")
