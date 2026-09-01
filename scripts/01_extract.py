"""Extract activations for every (language, arm, concept) sheet in data/raw/.

    python scripts/01_extract.py --cpu          # real model + real sheets
    python scripts/01_extract.py --synthetic    # fake, no model needed

Sheets live at data/raw/{lang}_{arm}.xlsx, each with one or both concept tabs.
Activations save to data/activations/{lang}_{arm}_{concept}.npz
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd  # noqa: E402

from prov.config import Config  # noqa: E402
from prov import data as D  # noqa: E402
from prov import extract as E  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--synthetic", action="store_true",
                help="fabricate activations to test the pipeline with no model")
ap.add_argument("--cpu", action="store_true", help="force CPU (no GPU present)")
ap.add_argument("--pool", default="mean", choices=["last", "mean"],
                help="read-out position; mean is better for high-fertility "
                     "languages, last for English-like ones")
ap.add_argument("--n-pairs", type=int, default=100, help="synthetic only")
args = ap.parse_args()

cfg = Config.load()
cfg.act_dir.mkdir(parents=True, exist_ok=True)
cfg.results_dir.mkdir(parents=True, exist_ok=True)
CONCEPTS = list(D.CONCEPT_SHEETS)

if args.synthetic:
    shifts = {"A": 0.0, "B": 0.20, "C": 0.60, "D": 0.55}
    for lang in cfg.languages:
        for arm in cfg.arms:
            for concept in CONCEPTS:
                pos, neg = E.synthetic(args.n_pairs, shift=shifts.get(arm, 0.0),
                                       seed=hash((lang, arm, concept)) % 10_000)
                df = pd.DataFrame({
                    "pair_id": [f"P{i:03d}" for i in range(len(pos))],
                    "writer_id": ["synthetic"] * len(pos),
                    "subject": [f"subj{i % 25}" for i in range(len(pos))],
                })
                E.save(cfg.act_dir / f"{lang}_{arm}_{concept}.npz", pos, neg, df)
                print(f"synthetic {lang} {arm} {concept}: {pos.shape}")
    sys.exit(0)

model, tok = E.load_model(cfg.model_id, cfg.dtype,
                          device="cpu" if args.cpu else "auto")

reports = []
for lang in cfg.languages:
    for arm in cfg.arms:
        path = cfg.raw_dir / f"{lang}_{arm}.xlsx"
        if not path.exists():
            continue

        for concept in D.available_concepts(path):
            try:
                df = D.load_sheet(path, lang=lang, arm=arm, concept=concept)
            except Exception as exc:
                print(f"skip {lang} {arm} {concept}: {exc}")
                continue
            rep = D.validate(df)
            print(f"\n=== {lang} {arm} {concept} [pool={args.pool}] ==="
                  f"\n{D.summarise(rep)}")
            if len(df) == 0:
                print(f"skip {lang} {arm} {concept}: no rows")
                continue
            reports.append(rep)

            pos, neg = E.extract_pairs(df, model, tok,
                                       batch_size=cfg.batch_size,
                                       max_length=cfg.max_length,
                                       pool=args.pool)
            out = cfg.act_dir / f"{lang}_{arm}_{concept}.npz"
            E.save(out, pos, neg, df)
            print(f"saved {pos.shape} -> {out.name}")

if not reports:
    sys.exit("no workbooks found in data/raw/ — expected e.g. yor_A.xlsx")

allrep = pd.concat(reports, ignore_index=True)
allrep.to_csv(cfg.results_dir / "data_quality.csv", index=False)

from prov.fertility import table  # noqa: E402
tbl = table(allrep, tok)
tbl.to_csv(cfg.results_dir / "fertility.csv", index=False)
print("\ntokenizer fertility (tokens per word):")
print(tbl.to_string(index=False))
