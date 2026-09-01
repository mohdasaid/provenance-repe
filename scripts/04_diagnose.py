"""Why is the noise floor low? Four hypotheses, tested separately.

    python scripts/04_diagnose.py --cpu

1. Is the concept linearly present at all?  -> cross-validated probe accuracy.
   If the probe works but diff-in-means does not, the concept IS there and the
   extraction method is the problem, not the model.
2. Is last-token the wrong read-out position? -> compare against mean pooling
   over real (non-pad) tokens. Short natural sentences with no prompt template
   often have an uninformative final token.
3. Is n too small? -> floor at n=25, 50, 100 to see whether it is climbing.
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
ap.add_argument("--tag", default="", help="suffix for the output file")
ap.add_argument("--max-overlap", type=float, default=0.15,
                help="length-overlap ceiling for layer selection")
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
subjects = df["subject"].to_numpy().astype(str) if "subject" in df.columns else None
lens = ((df["positive"].str.split().str.len()
         + df["negative"].str.split().str.len()) / 2).to_numpy(float)

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


def length_overlap(pos, neg, lengths):
    """|cos(concept direction, long-vs-short direction)| per layer.

    lengths must align with pos/neg row-for-row. Passing it in rather than
    closing over the full-sample vector matters for subsampled calls.
    """
    both = np.concatenate([pos, neg])
    ml = np.concatenate([lengths, lengths])
    med = np.median(lengths)
    ldir = both[ml > med].mean(0) - both[ml <= med].mean(0)
    ldir /= np.clip(np.linalg.norm(ldir, axis=-1, keepdims=True), 1e-9, None)
    return np.abs(V.cosine(V.diff_in_means(pos, neg), ldir))


def report(name, pos, neg, pool=None):
    if subjects is not None:
        floor = V.split_half_floor_clustered(pos, neg, subjects=subjects,
                                             n_splits=20, seed=0)
    else:
        floor = V.split_half_floor(pos, neg, n_splits=20, seed=0)

    # A bare argmax picks the most STABLE layer, and length-contaminated
    # layers are highly stable because length is a strong, reproducible
    # signal. Restrict to layers where the direction is not mostly length.
    ov = length_overlap(pos, neg, lens)
    clean = ov < args.max_overlap
    if clean.any():
        layer = int(np.argmax(np.where(clean, floor["mean"], -np.inf)))
    else:
        layer = V.best_layer(floor)
        print(f"  [{name}] no layer under overlap {args.max_overlap}")

    bare = int(np.argmax(floor["mean"]))
    print(f"{name:<28} floor {floor['mean'][layer]:.3f} "
          f"(L{layer}, 95% {floor['lo'][layer]:.3f}-{floor['hi'][layer]:.3f}, "
          f"overlap {ov[layer]:.3f}, {int(clean.sum())}/{len(clean)} clean)")
    if bare != layer:
        print(f"{'':<28} bare argmax would take L{bare}: "
              f"floor {floor['mean'][bare]:.3f}, overlap {ov[bare]:.3f}")

    rows.append({"lang": args.lang, "concept": args.concept,
                 "model": cfg.model_id, "measure": "floor", "pool": pool,
                 "label": name, "layer": layer,
                 "value": floor["mean"][layer],
                 "lo": floor["lo"][layer], "hi": floor["hi"][layer],
                 "length_overlap": float(ov[layer]),
                 "n_clean_layers": int(clean.sum()),
                 "n_layers": int(len(clean)),
                 "bare_argmax_layer": bare,
                 "bare_argmax_floor": float(floor["mean"][bare]),
                 "bare_argmax_overlap": float(ov[bare])})
    return floor, layer

def clean_argmax_floor(pos, neg, lengths, subjects_sub):
    """Floor at the most stable layer that is not length-contaminated."""
    f = (V.split_half_floor_clustered(pos, neg, subjects=subjects_sub,
                                      n_splits=20, seed=0)
         if subjects_sub is not None else
         V.split_half_floor(pos, neg, n_splits=20, seed=0))
    ov = length_overlap(pos, neg, lengths)
    cl = ov < args.max_overlap
    L = (int(np.argmax(np.where(cl, f["mean"], -np.inf))) if cl.any()
         else V.best_layer(f))
    return f["mean"][L], L, int(cl.sum())


rows = []
results = {}
for pool in ("last", "mean"):
    print(f"extracting [{pool}] ...")
    pos = extract_pooled(df["positive"].tolist(), pool)
    neg = extract_pooled(df["negative"].tolist(), pool)
    results[pool] = (pos, neg)

print("\n--- 1/2. read-out position, raw ---")
for pool, (pos, neg) in results.items():
    report(f"{pool}-token", pos, neg, pool=pool)

# Centring is not reported and not computed: diff_in_means subtracts one
# class mean from the other, so any constant removed from both cancels
# exactly. It was a no-op that read as if it mattered.

print("\n--- 1. is the concept linearly there at all? ---")
print("(cross-validated probe accuracy; 0.5 = chance)")
for pool, (pos, neg) in results.items():
    best = (0, -1)
    for layer in range(pos.shape[1]):
        X = np.concatenate([pos[:, layer, :], neg[:, layer, :]])
        y = np.concatenate([np.ones(len(pos)), np.zeros(len(neg))])
        clf = make_pipeline(StandardScaler(),
                            LogisticRegression(max_iter=2000))
        acc = cross_val_score(clf, X, y, cv=5).mean()
        if acc > best[0]:
            best = (acc, layer)
    print(f"  {pool}-token: best CV accuracy {best[0]:.3f} at layer {best[1]}")
    rows.append({"lang": args.lang, "concept": args.concept,
             "model": cfg.model_id, "measure": "probe", "pool": pool,
             "layer": best[1], "value": best[0]})

print("\n--- 3. is n the problem? ---")
print("  subjects sampled at random, 10 draws per size; spread across draws")
pos, neg = results["mean"]
rng = np.random.default_rng(0)
uniq = np.unique(subjects) if subjects is not None else None
sizes = [25, 50, len(pos)]

for n in sizes:
    if subjects is None or n >= len(pos):
        val, L, _ = clean_argmax_floor(pos, neg, lens, subjects)
        vals, npairs = [val], [len(pos)]
    else:
        # take whole subjects, not the first n rows: the sheet is ordered by
        # subject with several pairs each, so pos[:25] is ~6 subjects and
        # confounds sample size with subject count
        k = max(2, round(n / len(pos) * len(uniq)))
        vals, npairs = [], []
        for _ in range(10):
            keep = rng.choice(uniq, k, replace=False)
            idx = np.flatnonzero(np.isin(subjects, keep))
            if len(idx) < 10:
                continue
            v, _, _ = clean_argmax_floor(pos[idx], neg[idx], lens[idx],
                                         subjects[idx])
            vals.append(v)
            npairs.append(len(idx))
    if not vals:
        continue
    vals = np.array(vals)
    print(f"  n~{n:<4} floor {vals.mean():.3f} "
          f"(sd {vals.std():.3f}, range {vals.min():.3f}-{vals.max():.3f}, "
          f"{len(vals)} draws, {int(np.mean(npairs))} pairs)")
    rows.append({"lang": args.lang, "concept": args.concept,
                 "model": cfg.model_id, "measure": "n_scaling",
                 "pool": "mean", "n": n,
                 "n_pairs_actual": float(np.mean(npairs)),
                 "value": float(vals.mean()), "sd": float(vals.std()),
                 "lo": float(vals.min()), "hi": float(vals.max()),
                 "n_draws": len(vals)})

print("\n--- sanity ---")
pos, neg = results["last"]
print(f"  activation norm (mean): {np.linalg.norm(pos, axis=-1).mean():.1f}")
print(f"  any NaN: {bool(np.isnan(pos).any())}")

texts = df["positive"].tolist() + df["negative"].tolist()
tokens = sum(len(tok.encode(t, add_special_tokens=False)) for t in texts)
words = sum(len(t.split()) for t in texts)
fert = tokens / words
print(f"  tokenizer fertility: {fert:.3f} tokens/word "
      f"(both sides, {len(texts)} texts)")

rows.append({"lang": args.lang, "concept": args.concept,
             "model": cfg.model_id, "measure": "fertility",
             "value": fert, "n": len(texts)})

import pandas as pd
suffix = f"_{args.tag}" if args.tag else ""
cfg.results_dir.mkdir(parents=True, exist_ok=True)
out = cfg.results_dir / f"diagnostic_{args.lang}_{args.concept}{suffix}.csv"
pd.DataFrame(rows).to_csv(out, index=False)
print(f"\nwrote {out}")
