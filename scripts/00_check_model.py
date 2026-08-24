"""Day-one sanity check. Run this BEFORE anything else, before data exists.

Confirms the model loads, hidden_states behave like a normal residual stack,
and the last-token gather picks the token you think it does. Gemma 4's
E-variants and multimodal wrapper are exactly the kind of thing that can make
these assumptions quietly false.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch  # noqa: E402

from prov.config import Config  # noqa: E402
from prov.extract import load_model  # noqa: E402

cfg = Config.load()
print(f"loading {cfg.model_id} ...")
model, tok = load_model(cfg.model_id, cfg.dtype)

texts = ["short one", "a noticeably longer sentence than the first one here"]
enc = tok(texts, return_tensors="pt", padding=True)
enc = {k: v.to(model.device) for k, v in enc.items()}

with torch.no_grad():
    out = model(**enc, output_hidden_states=True)

hs = out.hidden_states
print(f"\nhidden_states entries: {len(hs)}  (expect n_layers + 1)")
print(f"config num_hidden_layers: {getattr(model.config, 'num_hidden_layers', '?')}")

shapes = {tuple(h.shape) for h in hs}
print(f"distinct shapes across layers: {shapes}")
if len(shapes) > 1:
    print("  !! layers differ in shape - NOT a uniform residual stack.")
    print("  !! per-layer directions will not be comparable. Fall back to a dense model.")
else:
    print("  ok: uniform residual stream")

mask = enc["attention_mask"]
last = mask.sum(1) - 1
print(f"\ntrue final-token index per sequence: {last.tolist()}")
print(f"padded sequence length: {mask.shape[1]}")

naive = hs[-1][:, -1, :]
correct = hs[-1][torch.arange(len(texts)), last, :]
delta = (naive - correct).abs().max().item()
print(f"max |naive last-position - mask-gathered|: {delta:.4f}")
print("  nonzero => right padding is active and the naive [:, -1, :] gather is WRONG")

print(f"\ndevice: {model.device} | dtype: {next(model.parameters()).dtype}")
if torch.cuda.is_available():
    print(f"VRAM allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
