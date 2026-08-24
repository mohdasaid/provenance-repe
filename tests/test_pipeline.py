"""The tests that matter: does the analysis find an effect when one is really
there, and stay quiet when it is not?

Run:  python -m pytest tests/ -v      (or just: python tests/test_pipeline.py)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from prov import extract as E
from prov import vectors as V


def test_null_case_stays_within_floor():
    """Two arms with the SAME underlying direction should not be flagged."""
    pos_a, neg_a = E.synthetic(100, shift=0.0, seed=1)
    pos_b, neg_b = E.synthetic(100, shift=0.0, seed=2)
    floor = V.split_half_floor(pos_a, neg_a, n_splits=20, seed=0)
    c = V.compare_arms((pos_a, neg_a), (pos_b, neg_b))
    flagged = V.verdict(c, floor)
    rate = flagged.mean()
    assert rate < 0.25, f"false-positive rate too high: {rate:.2f}"


def test_real_effect_is_detected():
    """An arm whose direction has genuinely rotated should fall below floor."""
    pos_a, neg_a = E.synthetic(100, shift=0.0, seed=1)
    pos_c, neg_c = E.synthetic(100, shift=0.60, seed=2)
    floor = V.split_half_floor(pos_a, neg_a, n_splits=20, seed=0)
    c = V.compare_arms((pos_a, neg_a), (pos_c, neg_c))
    flagged = V.verdict(c, floor)
    assert flagged.mean() > 0.75, f"detection rate too low: {flagged.mean():.2f}"


def test_floor_widens_when_n_is_small():
    """Fewer pairs => noisier direction => wider floor. This is why the
    30-pair pilot exists."""
    pos, neg = E.synthetic(100, seed=1)
    wide = V.split_half_floor(pos[:20], neg[:20], n_splits=20, seed=0)
    tight = V.split_half_floor(pos, neg, n_splits=20, seed=0)
    assert wide["mean"].mean() < tight["mean"].mean()


def test_length_confound_leaks_into_vector():
    """Sanity check on why pair matching matters: an artificial offset applied
    only to positives produces a direction unrelated to the concept."""
    pos, neg = E.synthetic(100, seed=1)
    clean = V.diff_in_means(pos, neg)
    bias = np.zeros_like(pos)
    bias[:, :, 0] = 60.0                     # stand-in for a systematic nuisance
    dirty = V.diff_in_means(pos + bias, neg)
    sim = V.cosine(clean, dirty)
    assert sim.mean() < 0.5, "nuisance signal should dominate the direction"


def test_probe_transfer_degrades_across_concepts():
    """Synthetic activations are linearly separable by construction, so a
    modest rotation still transfers near-perfectly. What must degrade is
    transfer to a genuinely DIFFERENT concept — that is the mechanical
    property the probe is measuring."""
    pos_a, neg_a = E.synthetic(150, seed=1, concept_seed=1234)
    pos_x, neg_x = E.synthetic(150, seed=2, concept_seed=9999)
    same = V.probe_transfer((pos_a, neg_a), (pos_a, neg_a), layer=12)
    cross = V.probe_transfer((pos_a, neg_a), (pos_x, neg_x), layer=12)
    assert same > 0.95
    assert cross < 0.75, f"different concept should not transfer: {cross:.2f}"


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"PASS {name}")
    print("\nall tests passed")
