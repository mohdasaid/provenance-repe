"""The whole result, from saved activations.

    python scripts/02_analyze.py

Produces results/noise_floor.csv, results/arm_cosines.csv,
results/probe_transfer.csv and a plain-text verdict per language.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from prov.config import Config  # noqa: E402
from prov import extract as E  # noqa: E402
from prov import vectors as V  # noqa: E402

cfg = Config.load()
cfg.results_dir.mkdir(parents=True, exist_ok=True)

floor_rows, cos_rows, probe_rows = [], [], []

for lang in cfg.languages:
    arms = {}
    for arm in cfg.arms:
        p = cfg.act_dir / f"{lang}_{arm}.npz"
        if p.exists():
            pos, neg, _, _ = E.load(p)
            arms[arm] = (pos, neg)

    if "A" not in arms:
        print(f"{lang}: no arm A, skipping")
        continue

    # --- the noise floor, from arm A against itself -----------------------
    floor = V.split_half_floor(*arms["A"], n_splits=cfg.n_splits, seed=cfg.seed)
    L = len(floor["mean"])
    for l in range(L):
        floor_rows.append({
            "lang": lang, "layer": l,
            "floor_mean": floor["mean"][l],
            "floor_lo": floor["lo"][l],
            "floor_hi": floor["hi"][l],
        })

    layer = V.best_layer(floor)
    print(f"\n=== {lang} ===")
    print(f"most stable layer: {layer}  "
          f"(split-half cosine {floor['mean'][layer]:.3f}, "
          f"95% range {floor['lo'][layer]:.3f}-{floor['hi'][layer]:.3f})")

    if floor["mean"][layer] < 0.7:
        print("  WARNING: even same-provenance halves disagree badly. "
              "n is too small or this concept is not linearly encoded here. "
              "No cross-arm comparison will mean anything until this improves.")

    # --- cross-arm comparison --------------------------------------------
    for arm, acts in arms.items():
        if arm == "A":
            continue
        c = V.compare_arms(arms["A"], acts)
        below = V.verdict(c, floor)
        for l in range(L):
            cos_rows.append({
                "lang": lang, "comparison": f"A_vs_{arm}", "layer": l,
                "cosine": c[l], "floor_lo": floor["lo"][l],
                "below_floor": bool(below[l]),
            })
        mark = "BELOW FLOOR" if below[layer] else "within noise"
        print(f"  A vs {arm} @layer{layer}: cos={c[layer]:.3f}  "
              f"floor_lo={floor['lo'][layer]:.3f}  -> {mark}")

        # --- probe transfer ----------------------------------------------
        acc_self = V.probe_transfer(arms["A"], arms["A"], layer, cfg.seed)
        acc_cross = V.probe_transfer(arms["A"], acts, layer, cfg.seed)
        probe_rows.append({
            "lang": lang, "train": "A", "test": arm, "layer": layer,
            "acc_within_A": acc_self, "acc_cross": acc_cross,
            "drop": acc_self - acc_cross,
        })
        print(f"       probe A->A {acc_self:.3f} | A->{arm} {acc_cross:.3f} "
              f"(drop {acc_self - acc_cross:+.3f})")

pd.DataFrame(floor_rows).to_csv(cfg.results_dir / "noise_floor.csv", index=False)
pd.DataFrame(cos_rows).to_csv(cfg.results_dir / "arm_cosines.csv", index=False)
pd.DataFrame(probe_rows).to_csv(cfg.results_dir / "probe_transfer.csv", index=False)
print(f"\nwrote 3 csvs to {cfg.results_dir}")
