"""Why is the noise floor low? Four hypotheses, tested separately.

    python scripts/04_diagnose.py --cpu

1. Is the concept linearly present at all?  -> cross-validated probe accuracy.
   If the probe works but diff-in-means does not, the concept IS there and the
   extraction method is the problem, not the model.
2. Is last-token the wrong read-out position? -> compare against mean pooling
   over real (non-pad) tokens. Short natural sentences with no prompt template
   often have an uninformative final token.
3. Are outlier dimensions swamping the difference? -> centre the activations
   (subtract the per-layer mean across all examples) before differencing.
4. Is n too small? -> floor at n=25, 50, 100 to see whether it is climbing.
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
print(f"{len(df)} pairs\n")

model, tok = E.load_model(cfg.model_id, cfg.dtype,
                          device="cpu" if args.cpu else "auto")


def extract_pooled(texts, pool):
    """pool: 'last' or 'mean' (over non-pad tokens)."""
    import torch
    chunks = []
    for i in range(0, len(texts), cfg.batch_size):
        batch = texts[i:i + cfg.batch_size]
        enc = tok(batch, return_tensors="pt", padding=True, truncation=True,
                  max_length=cfg.max_length)
        enc = {k: v.to(model.device) for k, v in enc.items()}
        with torch.no_grad():
            out = model(**enc, output_hidden_states=True)
        hs = torch.stack(out.hidden_states, dim=1)          # (B, L+1, T, d)
        mask = enc["attention_mask"]
        if pool == "last":
            last = mask.sum(1) - 1
            v = hs[torch.arange(hs.shape[0]), :, last, :]
        else:
            m = mask[:, None, :, None].float()
            v = (hs * m).sum(2) / m.sum(2).clamp(min=1)
        chunks.append(v.float().cpu().numpy())
        print(f"  {min(i + cfg.batch_size, len(texts))}/{len(texts)}", end="\r")
    print(" " * 30, end="\r")
    return np.concatenate(chunks, 0)


def report(name, pos, neg):
    floor = V.split_half_floor(pos, neg, n_splits=20, seed=0)
    layer = V.best_layer(floor)
    print(f"{name:<28} floor {floor['mean'][layer]:.3f} "
          f"(layer {layer}, 95% {floor['lo'][layer]:.3f}-{floor['hi'][layer]:.3f})")
    return floor, layer


results = {}
for pool in ("last", "mean"):
    print(f"extracting [{pool}] ...")
    pos = extract_pooled(df["positive"].tolist(), pool)
    neg = extract_pooled(df["negative"].tolist(), pool)
    results[pool] = (pos, neg)

print("\n--- 1/2. read-out position, raw ---")
for pool, (pos, neg) in results.items():
    report(f"{pool}-token", pos, neg)

print("\n--- 3. centred (per-layer mean removed) ---")
centred = {}
for pool, (pos, neg) in results.items():
    allx = np.concatenate([pos, neg], 0)
    mu = allx.mean(0, keepdims=True)
    cp, cn = pos - mu, neg - mu
    centred[pool] = (cp, cn)
    report(f"{pool}-token centred", cp, cn)

print("\n--- 1. is the concept linearly there at all? ---")
print("(cross-validated probe accuracy; 0.5 = chance)")
for pool, (pos, neg) in results.items():
    best = (0, -1)
    for layer in range(0, pos.shape[1], 4):
        X = np.concatenate([pos[:, layer, :], neg[:, layer, :]])
        y = np.concatenate([np.ones(len(pos)), np.zeros(len(neg))])
        clf = make_pipeline(StandardScaler(),
                            LogisticRegression(max_iter=2000))
        acc = cross_val_score(clf, X, y, cv=5).mean()
        if acc > best[0]:
            best = (acc, layer)
    print(f"  {pool}-token: best CV accuracy {best[0]:.3f} at layer {best[1]}")

print("\n--- 4. is n the problem? ---")
pos, neg = centred["mean"]
for n in (25, 50, len(pos)):
    f = V.split_half_floor(pos[:n], neg[:n], n_splits=20, seed=0)
    print(f"  n={n:<4} floor {f['mean'][V.best_layer(f)]:.3f}")

print("\n--- sanity ---")
pos, neg = results["last"]
print(f"  activation norm (mean): {np.linalg.norm(pos, axis=-1).mean():.1f}")
print(f"  any NaN: {bool(np.isnan(pos).any())}")
tokens = [len(tok.encode(t, add_special_tokens=False)) for t in df['positive']]
words = [len(t.split()) for t in df['positive']]
print(f"  tokenizer fertility: {sum(tokens) / sum(words):.2f} tokens/word")
