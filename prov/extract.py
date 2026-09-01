"""Extract pooled hidden states, one vector per layer per sentence.

Output shape: (n_sentences, n_layers + 1, d_model), float32 on disk.

The last-token gather is where this kind of pipeline usually goes wrong.
With right padding, position -1 is a PAD token for every sentence shorter than
the longest in the batch, so you silently extract the representation of padding.
We gather at (attention_mask.sum(1) - 1) instead, which is the true final token
of each sequence regardless of padding.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np


def load_model(model_id: str, dtype: str = "bfloat16", device: str = "auto"):
    """Returns (model, tokenizer). Imports live inside so the module can be
    imported without torch installed."""
    import torch
    from transformers import AutoModel, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"          # we gather by mask, so right is fine

    model = AutoModel.from_pretrained(
        model_id,
        dtype=getattr(torch, dtype),
        low_cpu_mem_usage=True,
        device_map=None if device == "cpu" else "auto",
    )
    model.eval()
    return model, tok


def extract(texts: list[str], model, tok, batch_size: int = 16,
            max_length: int = 128, pool: str = "last") -> np.ndarray:
    """Hidden state at every layer for every text.

    pool="last"  gather at the true final token (attention_mask.sum(1) - 1).
    pool="mean"  average over all real (non-pad) tokens.

    Which is better is language-dependent: the last-token advantage narrows
    as tokenizer fertility rises, and reverses in some cells. It is not a
    settled rule -- measure it per language and model. Pooling is done per
    layer so peak memory is one layer rather than the whole stack.
    """
    import torch

    if pool not in ("last", "mean"):
        raise ValueError(f"pool must be 'last' or 'mean', got {pool!r}")

    out_chunks = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        enc = tok(batch, return_tensors="pt", padding=True,
                  truncation=True, max_length=max_length)
        enc = {k: v.to(model.device) for k, v in enc.items()}

        with torch.no_grad():
            out = model(**enc, output_hidden_states=True)

        mask = enc["attention_mask"]
        if pool == "last":
            idx = mask.sum(1) - 1
            rows = torch.arange(mask.shape[0], device=mask.device)
            vecs = torch.stack([h[rows, idx, :] for h in out.hidden_states],
                               dim=1)
        else:
            m = mask[:, :, None].float()
            vecs = torch.stack([(h * m).sum(1) / m.sum(1).clamp(min=1)
                                for h in out.hidden_states], dim=1)

        assert vecs.shape[0] == len(batch), f"batch mismatch {vecs.shape}"
        assert vecs.shape[1] == len(out.hidden_states), "layer count mismatch"
        out_chunks.append(vecs.float().cpu().numpy())

        del out, vecs
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return np.concatenate(out_chunks, axis=0)


def extract_pairs(df, model, tok, **kw) -> tuple[np.ndarray, np.ndarray]:
    """Returns (pos_acts, neg_acts), each (n_pairs, n_layers+1, d_model)."""
    pos = extract(df["positive"].tolist(), model, tok, **kw)
    neg = extract(df["negative"].tolist(), model, tok, **kw)
    return pos, neg


def act_path(act_dir: Path, lang: str, arm: str, concept: str,
             pool: str) -> Path:
    """Where a set of activations lives.

    The pooling belongs in the filename. Activations are pooling-specific, so
    without it a --pool mean run overwrites the --pool last output in place
    and whatever ran last is what every downstream script reads, with no
    record of which.
    """
    return act_dir / f"{lang}_{arm}_{concept}_{pool}.npz"


def save(path: Path, pos, neg, df, model_id: str = "", pool: str = "") -> None:
    """Write activations plus enough provenance to identify them later.

    model_id and pool are stored inside the file as well as in its name, so a
    renamed or relocated file can still say what produced it.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, pos=pos, neg=neg,
                        pair_id=df["pair_id"].to_numpy().astype(str),
                        writer_id=df["writer_id"].to_numpy().astype(str),
                        subject=df["subject"].to_numpy().astype(str),
                        model_id=np.array(model_id),
                        pool=np.array(pool))


def load(path: Path):
    """Returns (pos, neg, pair_id, writer_id, subject)."""
    z = np.load(path, allow_pickle=False)
    subj = z["subject"] if "subject" in z else None
    return z["pos"], z["neg"], z["pair_id"], z["writer_id"], subj


def load_meta(path: Path) -> dict:
    """What produced this file. Empty strings for files written before the
    metadata was recorded, so callers should treat "" as unknown rather than
    as a mismatch."""
    z = np.load(path, allow_pickle=False)
    return {"model_id": str(z["model_id"]) if "model_id" in z else "",
            "pool": str(z["pool"]) if "pool" in z else "",
            "n_pairs": int(z["pos"].shape[0]),
            "n_layers": int(z["pos"].shape[1]),
            "d_model": int(z["pos"].shape[2])}


def check_meta(path: Path, expect_pool: str = "", expect_model: str = "",
               strict: bool = True) -> dict:
    """Guard against analysing activations from the wrong run.

    Returns the metadata. Raises on a pooling mismatch when strict, since
    that silently changes what is being measured; a model mismatch only
    warns, because the config is reset often enough that the file is usually
    the more reliable record.
    """
    meta = load_meta(path)
    if expect_pool and meta["pool"] and meta["pool"] != expect_pool:
        msg = (f"{path.name} was written with pool={meta['pool']!r}, "
               f"analysis expects {expect_pool!r}")
        if strict:
            raise ValueError(msg)
        print(f"  WARNING: {msg}")
    if expect_model and meta["model_id"] and meta["model_id"] != expect_model:
        print(f"  WARNING: {path.name} came from {meta['model_id']}, "
              f"config says {expect_model}")
    return meta


# ---------------------------------------------------------------- synthetic
def synthetic(n_pairs: int, n_layers: int = 25, d_model: int = 256,
              effect: float = 4.0, shift: float = 0.0,
              seed: int = 0, concept_seed: int = 1234
              ) -> tuple[np.ndarray, np.ndarray]:
    """Fake activations so the whole pipeline can be tested with no model.

    `effect`       strength of the concept signal, shared by every arm.
    `shift`        rotates the concept direction away from the true one. Set
                   it > 0 to simulate an arm whose direction has genuinely
                   moved (a real provenance effect); 0 simulates no effect.
    `seed`         per-arm sampling noise only.
    `concept_seed` fixed across arms — this is what makes "the same concept"
                   the same concept. Vary it only to simulate a different
                   concept entirely.
    """
    crng = np.random.default_rng(concept_seed)
    rng = np.random.default_rng(seed)
    concept = crng.normal(size=(n_layers, d_model))
    concept /= np.linalg.norm(concept, axis=1, keepdims=True)

    other = crng.normal(size=(n_layers, d_model))
    other -= (other * concept).sum(1, keepdims=True) * concept
    other /= np.linalg.norm(other, axis=1, keepdims=True)

    direction = concept + shift * other
    direction /= np.linalg.norm(direction, axis=1, keepdims=True)

    base = rng.normal(size=(n_pairs, n_layers, d_model))
    pos = base + effect * direction
    neg = rng.normal(size=(n_pairs, n_layers, d_model)) - effect * direction
    return pos.astype(np.float32), neg.astype(np.float32)
