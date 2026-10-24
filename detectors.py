from __future__ import annotations
import numpy as np
from scipy.stats import norm


def two_proportion_pvalue(a: np.ndarray, b: np.ndarray) -> float:
    """Two-sided pooled z-test p-value for two binary samples."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.size < 2 or b.size < 2:
        return 1.0
    pooled = (a.sum() + b.sum()) / (a.size + b.size)
    se = np.sqrt(max(pooled * (1.0 - pooled) * (1.0 / a.size + 1.0 / b.size), 0.0))
    if se == 0:
        return 1.0
    z = (a.mean() - b.mean()) / se
    return float(2 * norm.sf(abs(z)))


def error_rate_alarms(errors: np.ndarray, window_size: int, alpha: float) -> np.ndarray:
    """Compare adjacent non-overlapping windows; mark the later window on a significant change."""
    errors = np.asarray(errors, dtype=int)
    alarms = np.zeros(errors.size, dtype=int)
    w = window_size
    for end in range(2 * w, errors.size + 1, w):
        p = two_proportion_pvalue(errors[end - 2*w:end-w], errors[end-w:end])
        if p < alpha:
            alarms[end-w:end] = 1
    return alarms


def feature_mean_alarms(values: np.ndarray, window_size: int, alpha: float) -> np.ndarray:
    """Compare adjacent windows using a Welch-style normal approximation for mean change."""
    values = np.asarray(values, dtype=float)
    alarms = np.zeros(values.size, dtype=int)
    w = window_size
    for end in range(2 * w, values.size + 1, w):
        a, b = values[end - 2*w:end-w], values[end-w:end]
        se = np.sqrt(np.var(a, ddof=1)/len(a) + np.var(b, ddof=1)/len(b))
        if se > 0:
            z = abs(a.mean() - b.mean()) / se
            p = float(2 * norm.sf(z))
            if p < alpha:
                alarms[end-w:end] = 1
    return alarms


def page_hinkley_alarms(signal: np.ndarray, delta: float = 0.005, threshold: float = 5.0) -> np.ndarray:
    """Simple Page-Hinkley-style cumulative monitor for upward changes in a scalar signal."""
    x = np.asarray(signal, dtype=float)
    alarms = np.zeros(x.size, dtype=int)
    mean = 0.0
    cumulative = 0.0
    minimum = 0.0
    count = 0
    for i, value in enumerate(x):
        count += 1
        mean += (value - mean) / count
        cumulative += value - mean - delta
        minimum = min(minimum, cumulative)
        if cumulative - minimum > threshold:
            alarms[i] = 1
            cumulative = 0.0
            minimum = 0.0
            mean = 0.0
            count = 0
    return alarms
