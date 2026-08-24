import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from prov.config import Config
from prov import extract as E

cfg = Config.load()
model, tok = E.load_model(cfg.model_id, cfg.dtype, device="cpu")

texts = [f"sentence number {i} for testing" for i in range(20)]
acts = E.extract(texts, model, tok, batch_size=8, max_length=64)

print("shape:", acts.shape)          # expect (20, 36, 1536)
print("NaNs:", bool((acts != acts).any()))
print("dtype:", acts.dtype)