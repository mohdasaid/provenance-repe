"""Central config. Everything model- or path-specific lives here, not in scripts."""
from dataclasses import dataclass, field
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parent.parent


@dataclass
class Config:
    model_id: str = "google/gemma-4-E2B"
    dtype: str = "bfloat16"
    batch_size: int = 16
    max_length: int = 128
    n_layers: int = 35          # E2B: 35 layers, hidden 1536
    d_model: int = 1536

    languages: list = field(default_factory=lambda: ["hau", "yor"])
    arms: list = field(default_factory=lambda: ["A", "B", "C", "D"])

    n_splits: int = 20          # split-half repeats for the noise floor
    seed: int = 0

    raw_dir: Path = ROOT / "data" / "raw"
    act_dir: Path = ROOT / "data" / "activations"
    results_dir: Path = ROOT / "results"

    @classmethod
    def load(cls, path: Path | str | None = None) -> "Config":
        path = Path(path) if path else ROOT / "config.yaml"
        if not path.exists():
            return cls()
        with open(path) as f:
            raw = yaml.safe_load(f) or {}
        for k in ("raw_dir", "act_dir", "results_dir"):
            if k in raw:
                raw[k] = Path(raw[k])
        return cls(**raw)
