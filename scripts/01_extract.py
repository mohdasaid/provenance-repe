"""Extract activations for every (language, arm) sheet found in data/raw/.

    python scripts/01_extract.py                  # real model + real sheets
    python scripts/01_extract.py --synthetic      # fake everything, no GPU

Sheets are expected at data/raw/{lang}_{arm}.xlsx  e.g. hau_A.xlsx, yor_C.xlsx
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

ap = argparse.ArgumentParser()
ap.add_argument("--synthetic", action="store_true",
                help="fabricate activations to test the pipeline with no model")
ap.add_argument("--n-pairs", type=int, default=100, help="synthetic only")
args = ap.parse_args()

cfg = Config.load()
cfg.act_dir.mkdir(parents=True, exist_ok=True)
cfg.results_dir.mkdir(parents=True, exist_ok=True)

if args.synthetic:
    # arm A is the reference; C and D are given a rotated direction so you can
    # confirm the analysis actually detects an effect when one exists.
    shifts = {"A": 0.0, "B": 0.20, "C": 0.60, "D": 0.55}
    for lang in cfg.languages:
        for arm in cfg.arms:
            pos, neg = E.synthetic(args.n_pairs, shift=shifts.get(arm, 0.0),
                                   seed=hash((lang, arm)) % 10_000)
            df = pd.DataFrame({
                "pair_id": [f"P{i:03d}" for i in range(len(pos))],
                "writer_id": ["synthetic"] * len(pos),
            })
            E.save(cfg.act_dir / f"{lang}_{arm}.npz", pos, neg, df)
            print(f"synthetic {lang} {arm}: {pos.shape}")
    sys.exit(0)

model, tok = E.load_model(cfg.model_id, cfg.dtype)

reports = []
for lang in cfg.languages:
    for arm in cfg.arms:
        path = cfg.raw_dir / f"{lang}_{arm}.xlsx"
        if not path.exists():
            print(f"skip {path.name} (not present)")
            continue

        df = D.load_sheet(path, lang=lang, arm=arm)
        rep = D.validate(df)
        print(f"\n=== {lang} {arm} ===\n{D.summarise(rep)}")
        reports.append(rep)

        pos, neg = E.extract_pairs(df, model, tok,
                                   batch_size=cfg.batch_size,
                                   max_length=cfg.max_length)
        E.save(cfg.act_dir / f"{lang}_{arm}.npz", pos, neg, df)
        print(f"saved {pos.shape} -> {lang}_{arm}.npz")

if reports:
    allrep = pd.concat(reports, ignore_index=True)
    allrep.to_csv(cfg.results_dir / "data_quality.csv", index=False)

    from prov.fertility import table
    table(allrep, tok).to_csv(cfg.results_dir / "fertility.csv", index=False)
    print("\nwrote results/data_quality.csv and results/fertility.csv")
