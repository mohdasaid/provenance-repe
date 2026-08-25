"""Compare direction estimators on real activations.

    python scripts/05_estimators.py --lang yor --concept sentiment
    python scripts/05_estimators.py --lang eng --concept sentiment

Extracts once per pooling strategy, then compares diff-in-means against
mass-mean probing at several shrinkage levels. Shrinkage 1.0 reduces mass-mean
to diff-in-means up to scaling, so the sweep brackets the answer.

The probe accuracy is the ceiling: it is what a linear classifier can achieve
on the same data. If an estimator's floor is far below its probe accuracy,
the concept is present and the estimator is losing it.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.model_selection import cross_val_score  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from prov.config import Config  # noqa: E402
from prov import data as D  # noqa: E402
from prov import extract as E  # noqa: E402
from prov import vectors as V  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--cpu", action="store_true")
ap.add_argument("--lang", default="yor")
ap.add_argument("--concept", default="sentiment")
args = ap.parse_args()

cfg = Config.load()

if args.lang == "eng":
    import pandas as pd
    seeds = pd.read_csv(cfg.raw_dir / "english_seeds.csv")
    df = seeds[seeds["concept"] == args.concept].rename(columns={
        "positive_en": "positive", "negative_en": "negative"})
else:
    df = D.load_sheet(cfg.raw_dir / f"{args.lang}_A.xlsx", args.lang, "A",
                      concept=args.concept)

print(f"{args.lang} / {args.concept}: {len(df)} pairs")
print(f"model: {cfg.model_id}\n")

model, tok = E.load_model(cfg.model_id, cfg.dtype,
                          device="cpu" if args.cpu else "auto")


def extract_pooled(texts, pool):
    """Pool per layer so peak memory is one layer, not all of them."""
    import torch
    chunks = []
    for i in range(0, len(texts), cfg.batch_size):
        batch = texts[i:i + cfg.batch_size]
        enc = tok(batch, return_tensors="pt", padding=True, truncation=True,
                  max_length=cfg.max_length)
        enc = {k: v.to(model.device) for k, v in enc.items()}
        with torch.no_grad():
            out = model(**enc, output_hidden_states=True)
        mask = enc["attention_mask"]
        if pool == "last":
            idx = mask.sum(1) - 1
            v = torch.stack([h[torch.arange(h.shape[0]), idx, :]
                             for h in out.hidden_states], dim=1)
        else:
            m = mask[:, :, None].float()
            v = torch.stack([(h * m).sum(1) / m.sum(1).clamp(min=1)
                             for h in out.hidden_states], dim=1)
        chunks.append(v.float().cpu().numpy())
        del out, v
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(f"  {min(i + cfg.batch_size, len(texts))}/{len(texts)}", end="\r")
    print(" " * 30, end="\r")
    return np.concatenate(chunks, 0)


def probe_ceiling(pos, neg):
    """Best cross-validated accuracy across layers — the ceiling."""
    best = (0.0, -1)
    for layer in range(0, pos.shape[1], 2):
        X = np.concatenate([pos[:, layer, :], neg[:, layer, :]])
        y = np.concatenate([np.ones(len(pos)), np.zeros(len(neg))])
        clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
        acc = cross_val_score(clf, X, y, cv=5).mean()
        if acc > best[0]:
            best = (acc, layer)
    return best


results = {}
for pool in ("last", "mean"):
    print(f"extracting [{pool}] ...")
    pos = extract_pooled(df["positive"].tolist(), pool)
    neg = extract_pooled(df["negative"].tolist(), pool)
    results[pool] = (pos, neg)

print(f"\nactivation shape: {results['mean'][0].shape}\n")

print("probe ceiling (best CV accuracy over layers)")
for pool, (pos, neg) in results.items():
    acc, layer = probe_ceiling(pos, neg)
    print(f"  {pool:<5} {acc:.3f} at layer {layer}")

print("\nsplit-half floor by estimator (best layer, 95% interval)")
print(f"{'pooling':<8} {'estimator':<26} {'floor':>6}  {'95% interval':>16}")
print("-" * 60)

rows = []
for pool, (pos, neg) in results.items():
    configs = [("diff_in_means", {}), ("normalised_diff", {})]
    for est, kw in configs:
        f = V.split_half_floor_est(pos, neg, estimator=est,
                                   n_splits=cfg.n_splits, seed=cfg.seed, **kw)
        layer = int(np.argmax(f["mean"]))
        label = est if not kw else f"{est} (shrink {kw['shrinkage']})"
        print(f"{pool:<8} {label:<26} {f['mean'][layer]:>6.3f}  "
              f"{f['lo'][layer]:>7.3f}-{f['hi'][layer]:.3f}  L{layer}")
        rows.append({"lang": args.lang, "concept": args.concept, "pool": pool,
                     "estimator": label, "layer": layer,
                     "floor": f["mean"][layer], "lo": f["lo"][layer],
                     "hi": f["hi"][layer]})

import pandas as pd  # noqa: E402
out = cfg.results_dir / f"estimators_{args.lang}_{args.concept}.csv"
cfg.results_dir.mkdir(parents=True, exist_ok=True)
pd.DataFrame(rows).to_csv(out, index=False)
print(f"\nwrote {out}")
print("\nIf mass-mean beats diff-in-means at some shrinkage, the concept was")
print("there and the standard estimator was losing it. If neither approaches")
print("the probe ceiling, the direction is genuinely unstable at this n.")
