"""Extract last-token hidden states, one vector per layer per sentence.

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
import torch
from transformers import AutoModel, AutoTokenizer

def load_model(model_id: str, dtype: str = "bfloat16", device: str = "auto"):
    """Returns (model, tokenizer). Kept separate so scripts can be imported
    without torch installed."""
    

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
            max_length: int = 128) -> np.ndarray:
    """Last-token hidden state at every layer for every text."""
    import torch

    out_chunks = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        enc = tok(batch, return_tensors="pt", padding=True,
                  truncation=True, max_length=max_length)
        enc = {k: v.to(model.device) for k, v in enc.items()}

        with torch.no_grad():
            out = model(**enc, output_hidden_states=True)

        hs = torch.stack(out.hidden_states, dim=1)        # (B, L+1, T, d)
        last = enc["attention_mask"].sum(dim=1) - 1       # (B,) true final token
        idx = torch.arange(hs.shape[0], device=hs.device)
        vecs = hs[idx, :, last, :]                        # (B, L+1, d)

        assert vecs.shape[0] == len(batch), f"batch mismatch {vecs.shape}"
        assert vecs.shape[1] == len(out.hidden_states), "layer count mismatch"
        out_chunks.append(vecs.float().cpu().numpy())

        del out, hs, vecs
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return np.concatenate(out_chunks, axis=0)


def extract_pairs(df, model, tok, **kw) -> tuple[np.ndarray, np.ndarray]:
    """Returns (pos_acts, neg_acts), each (n_pairs, n_layers+1, d_model)."""
    pos = extract(df["positive"].tolist(), model, tok, **kw)
    neg = extract(df["negative"].tolist(), model, tok, **kw)
    return pos, neg


def save(path: Path, pos: np.ndarray, neg: np.ndarray, df) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, pos=pos, neg=neg,
                        pair_id=df["pair_id"].to_numpy().astype(str),
                        writer_id=df["writer_id"].to_numpy().astype(str))


def load(path: Path):
    z = np.load(path, allow_pickle=False)
    return z["pos"], z["neg"], z["pair_id"], z["writer_id"]


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
