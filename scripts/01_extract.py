"""Extract activations for every (language, arm, concept) sheet in data/raw/.

    python scripts/01_extract.py --cpu          # real model + real sheets
    python scripts/01_extract.py --synthetic    # fake, no model needed

Sheets live at data/raw/{lang}_{arm}.xlsx, each with one or both concept tabs.
Activations save to data/activations/{lang}_{arm}_{concept}_{pool}.npz

The pooling is in the filename because activations are pooling-specific: a
run with --pool mean would otherwise overwrite the --pool last output in
place, leaving whichever ran last on disk with no record of which. The model
id and pooling are also written inside each npz.
"""
import argparse
import hashlib
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
                help="read-out position; the last-token advantage narrows as "
                     "tokenizer fertility rises and reverses in some cells")
ap.add_argument("--n-pairs", type=int, default=100, help="synthetic only")
args = ap.parse_args()

cfg = Config.load()
cfg.act_dir.mkdir(parents=True, exist_ok=True)
cfg.results_dir.mkdir(parents=True, exist_ok=True)
CONCEPTS = list(D.CONCEPT_SHEETS)      # used by the synthetic branch
PAIRS_PER_SUBJECT = 4                  # matches the collection sheets

if args.synthetic:
    shifts = {"A": 0.0, "B": 0.20, "C": 0.60, "D": 0.55}
    for lang in cfg.languages:
        for arm in cfg.arms:
            for concept in CONCEPTS:
                # md5, not hash(): Python randomises string hashes per
                # process, so hash() makes synthetic runs unreproducible
                seed = int(hashlib.md5(
                    f"{lang}_{arm}_{concept}".encode()).hexdigest()[:8],
                    16) % 10_000
                pos, neg = E.synthetic(args.n_pairs, shift=shifts.get(arm, 0.0),
                                       seed=seed)
                # group by a fixed pairs-per-subject so the clustering the
                # synthetic data exists to exercise survives any --n-pairs
                df = pd.DataFrame({
                    "pair_id": [f"P{i:03d}" for i in range(len(pos))],
                    "writer_id": ["synthetic"] * len(pos),
                    "subject": [f"subj{i // PAIRS_PER_SUBJECT:03d}"
                                for i in range(len(pos))],
                })
                out = E.act_path(cfg.act_dir, lang, arm, concept, args.pool)
                E.save(out, pos, neg, df, model_id="synthetic",
                       pool=args.pool)
                print(f"synthetic {lang} {arm} {concept}: {pos.shape} "
                      f"-> {out.name}")
    sys.exit(0)

model, tok = E.load_model(cfg.model_id, cfg.dtype,
                          device="cpu" if args.cpu else "auto")

reports, skipped = [], []
for lang in cfg.languages:
    for arm in cfg.arms:
        path = cfg.raw_dir / f"{lang}_{arm}.xlsx"
        if not path.exists():
            continue

        for concept in D.available_concepts(path):
            try:
                df = D.load_sheet(path, lang=lang, arm=arm, concept=concept)
            except Exception as exc:
                msg = f"{lang} {arm} {concept}: {exc}"
                print(f"skip {msg}")
                skipped.append(msg)
                continue
            rep = D.validate(df)
            print(f"\n=== {lang} {arm} {concept} [pool={args.pool}] ==="
                  f"\n{D.summarise(rep)}")
            if len(df) == 0:
                msg = f"{lang} {arm} {concept}: no rows"
                print(f"skip {msg}")
                skipped.append(msg)
                continue
            reports.append(rep)

            pos, neg = E.extract_pairs(df, model, tok,
                                       batch_size=cfg.batch_size,
                                       max_length=cfg.max_length,
                                       pool=args.pool)
            out = E.act_path(cfg.act_dir, lang, arm, concept, args.pool)
            E.save(out, pos, neg, df, model_id=cfg.model_id, pool=args.pool)
            print(f"saved {pos.shape} -> {out.name}")

if skipped:
    print(f"\n{len(skipped)} arm/concept combinations skipped:")
    for msg in skipped:
        print(f"  {msg}")
    print("  these contribute no rows to the quality or fertility tables, so")
    print("  a skipped arm and an arm that was never configured look alike")
    print("  in the outputs — the count above is the only record.")

if not reports:
    sys.exit("no workbooks found in data/raw/ — expected e.g. yor_A.xlsx")

allrep = pd.concat(reports, ignore_index=True)
allrep.to_csv(cfg.results_dir / f"data_quality_{args.pool}.csv", index=False)

from prov.fertility import table  # noqa: E402
tbl = table(allrep, tok)
tbl["n_pairs"] = tbl.apply(
    lambda r: int(((allrep.lang == r.lang) & (allrep.arm == r.arm)
                   & (allrep.concept == r.concept)).sum())
    if "concept" in tbl.columns else
    int(((allrep.lang == r.lang) & (allrep.arm == r.arm)).sum()), axis=1)
tbl.to_csv(cfg.results_dir / f"fertility_{args.pool}.csv", index=False)
print("\ntokenizer fertility (tokens per word)")
print(f"  arms that loaded: {len(reports)}; skipped: {len(skipped)}")
print(tbl.to_string(index=False))
