from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, brier_score_loss


def prediction_metrics(y: np.ndarray, prediction: np.ndarray, probability: np.ndarray,
                       confidence_threshold: float = 0.7) -> dict:
    y = np.asarray(y, dtype=int)
    prediction = np.asarray(prediction, dtype=int)
    probability = np.asarray(probability, dtype=float)
    confidence = np.maximum(probability, 1 - probability)
    accepted = confidence >= confidence_threshold
    selective_risk = float(np.mean(prediction[accepted] != y[accepted])) if accepted.any() else None
    return {
        "accuracy": float(accuracy_score(y, prediction)),
        "f1": float(f1_score(y, prediction, zero_division=0)),
        "roc_auc": float(roc_auc_score(y, probability)) if np.unique(y).size == 2 else None,
        "brier_score": float(brier_score_loss(y, probability)),
        "coverage": float(accepted.mean()),
        "selective_risk": selective_risk,
    }


def detection_metrics(alarms: np.ndarray, test_times: np.ndarray, shift_point: int,
                      tolerance: int) -> dict:
    alarms = np.asarray(alarms, dtype=int)
    test_times = np.asarray(test_times, dtype=int)
    alarm_times = test_times[alarms == 1]
    in_detection_region = (alarm_times >= shift_point) & (alarm_times <= shift_point + tolerance)
    valid = alarm_times[in_detection_region]
    false_alarms = alarm_times[(alarm_times < shift_point) | (alarm_times > shift_point + tolerance)]
    first = int(valid.min()) if valid.size else None
    return {
        "first_detection_time": first,
        "detection_delay": int(first - shift_point) if first is not None else None,
        "detected_within_tolerance": bool(first is not None),
        "false_alarm_count": int(false_alarms.size),
        "alarm_count": int(alarm_times.size),
    }


def aggregate_with_ci(frame: pd.DataFrame, group_columns: list[str], metric_columns: list[str]) -> pd.DataFrame:
    """Mean and approximate 95% CI across independent experiment runs."""
    rows = []
    for keys, group in frame.groupby(group_columns, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(group_columns, keys))
        for metric in metric_columns:
            vals = pd.to_numeric(group[metric], errors="coerce").dropna().to_numpy(dtype=float)
            row[f"{metric}_mean"] = float(vals.mean()) if vals.size else None
            half = 1.96 * vals.std(ddof=1) / np.sqrt(vals.size) if vals.size > 1 else None
            row[f"{metric}_ci95_halfwidth"] = float(half) if half is not None else None
            row[f"{metric}_n"] = int(vals.size)
        rows.append(row)
    return pd.DataFrame(rows)
