from __future__ import annotations
import numpy as np
import pandas as pd

FEATURES = ["x1", "x2", "x3", "x4"]
REGIMES = ("stationary", "covariate_shift", "concept_drift")


def make_stream(n_samples: int, shift_point: int, seed: int, regime: str) -> pd.DataFrame:
    """Generate a chronological binary-classification stream with known ground truth."""
    if regime not in REGIMES:
        raise ValueError(f"regime must be one of {REGIMES}")
    if n_samples < 500 or not 100 <= shift_point < n_samples - 100:
        raise ValueError("Require n_samples >= 500 and shift_point at least 100 from either end.")
    rng = np.random.default_rng(seed)
    time = np.arange(n_samples)
    post = time >= shift_point
    x1 = rng.normal(size=n_samples)
    x2 = rng.normal(size=n_samples)
    x3 = rng.normal(loc=(0.7 * post if regime == "covariate_shift" else np.zeros(n_samples)), size=n_samples)
    x4 = rng.normal(size=n_samples)

    base = 1.5 * x1 - 1.0 * x2 + 0.3 * x3 - 0.25 * x4
    if regime == "concept_drift":
        changed = -1.15 * x1 + 1.25 * x2 + 0.2 * x3 + 0.65 * x4
        logits = np.where(post, changed, base)
    else:
        logits = base
    p = 1.0 / (1.0 + np.exp(-logits))
    y = rng.binomial(1, p)
    return pd.DataFrame({
        "time": time, "x1": x1, "x2": x2, "x3": x3, "x4": x4,
        "y": y, "true_post_change": post.astype(int), "true_probability": p,
        "regime": regime,
    })
