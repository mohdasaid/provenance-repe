"""Concept directions, the split-half noise floor, and cross-arm comparison."""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

def normalised_diff(pos: np.ndarray, neg: np.ndarray,
                    normalise: bool = True) -> np.ndarray:
    """Unit-normalise each pair's difference before averaging, so one
    high-magnitude pair cannot dominate the direction."""
    d = pos - neg                                   # (n_pairs, L, d_model)
    d = d / np.clip(np.linalg.norm(d, axis=-1, keepdims=True), 1e-9, None)
    v = d.mean(axis=0)
    if normalise:
        v = v / np.clip(np.linalg.norm(v, axis=-1, keepdims=True), 1e-9, None)
    return v
    
def diff_in_means(pos: np.ndarray, neg: np.ndarray,
                  normalise: bool = True) -> np.ndarray:
    """Concept direction per layer. (n_pairs, L, d) -> (L, d)."""
    v = pos.mean(axis=0) - neg.mean(axis=0)
    if normalise:
        v = v / np.clip(np.linalg.norm(v, axis=-1, keepdims=True), 1e-9, None)
    return v


def cosine(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Per-layer cosine between two (L, d) direction stacks -> (L,)."""
    num = (a * b).sum(-1)
    den = np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1)
    return num / np.clip(den, 1e-9, None)


def split_half_floor(pos: np.ndarray, neg: np.ndarray, n_splits: int = 20,
                     seed: int = 0) -> dict:
    """The noise floor.

    Split one arm's pairs in half at random, build a direction from each half,
    and measure the cosine between them. Both halves are the same provenance,
    same writers, same subjects — so whatever spread you see here is pure
    sampling noise. Every cross-arm comparison must be read against it.
    """
    rng = np.random.default_rng(seed)
    n = pos.shape[0]
    cosines = []
    for _ in range(n_splits):
        idx = rng.permutation(n)
        a, b = idx[: n // 2], idx[n // 2:]
        va = diff_in_means(pos[a], neg[a])
        vb = diff_in_means(pos[b], neg[b])
        cosines.append(cosine(va, vb))
    c = np.stack(cosines)                       # (n_splits, L)
    return {
        "cosines": c,
        "mean": c.mean(0),
        "lo": np.percentile(c, 2.5, axis=0),
        "hi": np.percentile(c, 97.5, axis=0),
    }


def compare_arms(arm_x: tuple[np.ndarray, np.ndarray],
                 arm_y: tuple[np.ndarray, np.ndarray]) -> np.ndarray:
    """Per-layer cosine between the directions of two arms."""
    vx = diff_in_means(*arm_x)
    vy = diff_in_means(*arm_y)
    return cosine(vx, vy)


def verdict(observed: np.ndarray, floor: dict) -> np.ndarray:
    """True where the cross-arm cosine falls below the noise floor's 2.5th
    percentile — i.e. further apart than sampling noise can explain."""
    return observed < floor["lo"]


def probe_transfer(train: tuple[np.ndarray, np.ndarray],
                   test: tuple[np.ndarray, np.ndarray],
                   layer: int, seed: int = 0) -> float:
    """Train a linear probe on one arm, test on another. Accuracy on the
    held-out arm. If the concept is encoded the same way in both, transfer is
    high; if provenance moved it, transfer degrades."""
    ptr, ntr = train
    pte, nte = test
    Xtr = np.concatenate([ptr[:, layer, :], ntr[:, layer, :]])
    ytr = np.concatenate([np.ones(len(ptr)), np.zeros(len(ntr))])
    Xte = np.concatenate([pte[:, layer, :], nte[:, layer, :]])
    yte = np.concatenate([np.ones(len(pte)), np.zeros(len(nte))])

    clf = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=2000, random_state=seed),
    )
    clf.fit(Xtr, ytr)
    return float(clf.score(Xte, yte))


def best_layer(floor: dict) -> int:
    """A defensible default layer: the one where the direction is most stable
    under resampling. Report the full per-layer curve anyway."""
    return int(np.argmax(floor["mean"]))


# ---------------------------------------------------------------- estimators
def mass_mean(pos: np.ndarray, neg: np.ndarray, shrinkage: float = 0.1,
              normalise: bool = True) -> np.ndarray:
    """Mass-mean probing direction, per layer.

    Diff-in-means ignores how activations are spread out, so naturally
    high-variance directions dominate the difference even when they carry no
    information about the concept. Mass-mean whitens by the inverse covariance
    of the (class-centred) activations, which measures the difference in units
    of how much things normally vary in each direction.

    `shrinkage` blends the empirical covariance toward a scaled identity. With
    d_model >> n_pairs the empirical covariance is singular, so this is not
    optional - it is what makes the inverse exist at all.
    """
    n_layers, d = pos.shape[1], pos.shape[2]
    out = np.zeros((n_layers, d), dtype=np.float64)

    for l in range(n_layers):
        p, n = pos[:, l, :].astype(np.float64), neg[:, l, :].astype(np.float64)
        delta = p.mean(0) - n.mean(0)

        # within-class centring: remove each class's own mean before pooling,
        # so the covariance describes spread, not the between-class difference
        x = np.concatenate([p - p.mean(0), n - n.mean(0)], axis=0)
        cov = np.cov(x, rowvar=False)

        trace_scale = np.trace(cov) / d
        cov = (1 - shrinkage) * cov + shrinkage * trace_scale * np.eye(d)

        try:
            out[l] = np.linalg.solve(cov, delta)
        except np.linalg.LinAlgError:
            out[l] = np.linalg.pinv(cov) @ delta

    if normalise:
        out = out / np.clip(np.linalg.norm(out, axis=-1, keepdims=True), 1e-9, None)
    return out.astype(np.float32)


ESTIMATORS = {
    "diff_in_means": diff_in_means,
    "normalised_diff": normalised_diff,
}


def split_half_floor_est(pos, neg, estimator="diff_in_means", n_splits=20,
                         seed=0, **kw) -> dict:
    """split_half_floor, but with a choice of direction estimator."""
    fn = ESTIMATORS[estimator]
    rng = np.random.default_rng(seed)
    n = pos.shape[0]
    cosines = []
    for _ in range(n_splits):
        idx = rng.permutation(n)
        a, b = idx[: n // 2], idx[n // 2:]
        cosines.append(cosine(fn(pos[a], neg[a], **kw),
                              fn(pos[b], neg[b], **kw)))
    c = np.stack(cosines)
    return {"cosines": c, "mean": c.mean(0),
            "lo": np.percentile(c, 2.5, axis=0),
            "hi": np.percentile(c, 97.5, axis=0)}

def split_half_floor_clustered(pos, neg, subjects, n_splits=20, seed=0):
    """Resample whole subjects. With 4 pairs per subject, resampling pairs
    puts the same subject in both halves and inflates the floor."""
    rng = np.random.default_rng(seed)
    subs = np.unique(subjects)
    cosines = []
    for _ in range(n_splits):
        perm = rng.permutation(subs)
        s1 = set(perm[: len(perm) // 2])
        m = np.array([s in s1 for s in subjects])
        if m.sum() < 5 or (~m).sum() < 5:
            continue
        cosines.append(cosine(diff_in_means(pos[m], neg[m]),
                              diff_in_means(pos[~m], neg[~m])))
    c = np.stack(cosines)
    return {"cosines": c, "mean": c.mean(0),
            "lo": np.percentile(c, 2.5, axis=0),
            "hi": np.percentile(c, 97.5, axis=0)}
