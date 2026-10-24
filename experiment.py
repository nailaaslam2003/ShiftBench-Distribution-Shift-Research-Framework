from __future__ import annotations
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .data import make_stream, FEATURES, REGIMES
from .detectors import error_rate_alarms, feature_mean_alarms, page_hinkley_alarms
from .metrics import prediction_metrics, detection_metrics, aggregate_with_ci


def run(config: dict) -> dict:
    out = Path(config.get("output_dir", "reports"))
    out.mkdir(parents=True, exist_ok=True)
    n, shift = int(config["n_samples"]), int(config["shift_point"])
    train_end = min(int(n * float(config["train_fraction"])), shift)
    window, alpha = int(config["window_size"]), float(config["alpha"])
    seeds = range(int(config["seed_start"]), int(config["seed_start"]) + int(config["n_seeds"]))
    tolerance = int(config["detection_tolerance"])
    rows = []
    window_rows = []

    for regime in REGIMES:
        for seed in seeds:
            stream = make_stream(n, shift, seed, regime)
            train = stream.iloc[:train_end]
            test = stream.iloc[train_end:].copy()
            model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, class_weight="balanced", random_state=seed))
            model.fit(train[FEATURES], train["y"])
            prob = model.predict_proba(test[FEATURES])[:, 1]
            pred = (prob >= 0.5).astype(int)
            errors = (pred != test["y"].to_numpy()).astype(int)
            test_times = test["time"].to_numpy()

            detector_outputs = {
                "error_rate": error_rate_alarms(errors, window, alpha),
                "feature_mean_x3": feature_mean_alarms(test["x3"].to_numpy(), window, alpha),
                "page_hinkley_error": page_hinkley_alarms(errors.astype(float), threshold=5.0),
            }
            pm = prediction_metrics(test["y"].to_numpy(), pred, prob, float(config["confidence_threshold"]))
            for detector, alarms in detector_outputs.items():
                dm = detection_metrics(alarms, test_times, shift, tolerance)
                rows.append({
                    "regime": regime, "seed": seed, "detector": detector,
                    **pm, **dm,
                    "pre_change_error_rate": float(errors[test_times < shift].mean()) if np.any(test_times < shift) else None,
                    "post_change_error_rate": float(errors[test_times >= shift].mean()) if np.any(test_times >= shift) else None,
                })

            # Store per-window metrics for the error monitor, aiding visual diagnosis.
            for start in range(0, len(test), window):
                chunk = test.iloc[start:start+window]
                idx = chunk.index.to_numpy() - test.index[0]
                window_rows.append({
                    "regime": regime, "seed": seed,
                    "start_time": int(chunk["time"].iloc[0]),
                    "end_time": int(chunk["time"].iloc[-1]),
                    "error_rate": float(errors[idx].mean()),
                    "mean_confidence": float(np.maximum(prob[idx], 1-prob[idx]).mean()),
                })

    runs = pd.DataFrame(rows)
    runs.to_csv(out / "run_level_results.csv", index=False)
    windows = pd.DataFrame(window_rows)
    windows.to_csv(out / "window_metrics.csv", index=False)
    aggregate = aggregate_with_ci(
        runs, ["regime", "detector"],
        ["accuracy", "f1", "roc_auc", "brier_score", "coverage", "selective_risk",
         "detection_delay", "false_alarm_count", "detected_within_tolerance"]
    )
    aggregate.to_csv(out / "aggregate_results.csv", index=False)

    # Sensitivity sweep on detector windows and significance thresholds, using one seed per regime.
    sensitivity = []
    for regime in REGIMES:
        stream = make_stream(n, shift, int(config["seed_start"]), regime)
        train = stream.iloc[:train_end]
        test = stream.iloc[train_end:]
        model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, class_weight="balanced"))
        model.fit(train[FEATURES], train["y"])
        prob = model.predict_proba(test[FEATURES])[:, 1]
        errors = ((prob >= 0.5).astype(int) != test["y"].to_numpy()).astype(int)
        times = test["time"].to_numpy()
        for w in sorted(set([max(20, window//2), window, window*2])):
            if w * 2 > len(test):
                continue
            for a in [0.001, 0.01, 0.05]:
                alarms = error_rate_alarms(errors, w, a)
                dm = detection_metrics(alarms, times, shift, tolerance)
                sensitivity.append({"regime": regime, "window_size": w, "alpha": a, **dm})
    sens = pd.DataFrame(sensitivity)
    sens.to_csv(out / "sensitivity_results.csv", index=False)

    # Figures.
    fig, ax = plt.subplots(figsize=(9, 5))
    subset = aggregate[aggregate["detector"] == "error_rate"]
    for regime in REGIMES:
        r = subset[subset["regime"] == regime]
        if not r.empty:
            ax.bar(regime, float(r["f1_mean"].iloc[0]), label=regime)
    ax.set(title="Mean predictive F1 by stream regime", ylabel="F1 score", ylim=(0, 1))
    ax.tick_params(axis="x", rotation=15)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "performance_by_regime.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5))
    pivot = aggregate.pivot(index="detector", columns="regime", values="false_alarm_count_mean")
    pivot.plot(kind="bar", ax=ax)
    ax.set(title="False alarms by detector and regime", ylabel="Mean false-alarm count")
    ax.tick_params(axis="x", rotation=15)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "detector_comparison.png", dpi=160)
    plt.close(fig)

    # Selective risk-coverage curve on one reproducible concept-drift stream.
    stream = make_stream(n, shift, int(config["seed_start"]), "concept_drift")
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, class_weight="balanced"))
    model.fit(stream.iloc[:train_end][FEATURES], stream.iloc[:train_end]["y"])
    test = stream.iloc[train_end:]
    prob = model.predict_proba(test[FEATURES])[:, 1]
    conf = np.maximum(prob, 1-prob)
    pred = (prob >= .5).astype(int)
    curve = []
    for threshold in np.linspace(.5, .95, 19):
        accepted = conf >= threshold
        curve.append({"threshold": threshold, "coverage": accepted.mean(),
                      "selective_risk": np.mean(pred[accepted] != test["y"].to_numpy()[accepted]) if accepted.any() else np.nan})
    curve_df = pd.DataFrame(curve)
    curve_df.to_csv(out / "selective_curve.csv", index=False)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(curve_df["coverage"], curve_df["selective_risk"], marker="o")
    ax.set(title="Selective prediction on concept drift", xlabel="Coverage", ylabel="Error among accepted predictions")
    ax.grid(True, alpha=.3)
    fig.tight_layout()
    fig.savefig(out / "selective_risk_coverage.png", dpi=160)
    plt.close(fig)

    summary = {
        "project": "ShiftBench",
        "config": config,
        "runs": int(len(runs)),
        "regimes": list(REGIMES),
        "detectors": sorted(runs["detector"].unique().tolist()),
        "outputs": ["run_level_results.csv", "aggregate_results.csv", "sensitivity_results.csv",
                    "window_metrics.csv", "selective_curve.csv", "performance_by_regime.png",
                    "detector_comparison.png", "selective_risk_coverage.png"],
        "interpretation_warning": "Synthetic benchmark only. Approximate confidence intervals and simple detectors; no real-world or novelty claim.",
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
