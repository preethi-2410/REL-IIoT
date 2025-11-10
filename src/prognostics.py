"""
prognostics.py
Adds predictive maintenance analytics to anomaly results.
Computes severity scores, estimates Remaining Useful Life (RUL), 
and assigns health states for each motor.
"""

import os
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)


def compute_severity(error_series):
    """
    Normalize reconstruction errors between 0 and 1 to represent severity.
    """
    e_min, e_max = np.min(error_series), np.max(error_series)
    return (error_series - e_min) / (e_max - e_min + 1e-6)


def estimate_rul(errors, sampling_interval=1.0, failure_threshold=0.8):
    """
    Estimate Remaining Useful Life (RUL) based on degradation trend.
    Assumes linear degradation in severity.
    sampling_interval: time per data point (in hours or minutes)
    failure_threshold: severity threshold for failure
    """
    severity = compute_severity(errors)
    smoothed = savgol_filter(severity, 101, 2, mode="interp")  # smooth the trend

    # compute slope of degradation
    t = np.arange(len(smoothed))
    slope, intercept = np.polyfit(t, smoothed, 1)

    if slope <= 0:
        return np.inf  # no degradation detected

    # when will severity reach failure threshold
    current_severity = smoothed[-1]
    steps_to_failure = (failure_threshold - current_severity) / slope
    if steps_to_failure < 0:
        steps_to_failure = 0

    estimated_rul = steps_to_failure * sampling_interval
    return estimated_rul


def add_prognostics_to_results(file_path, sampling_interval=1.0):
    """
    Load anomaly results, compute severity and RUL, and save extended file.
    """
    if not os.path.exists(file_path):
        print(f"[ERROR] File not found: {file_path}")
        return

    df = pd.read_csv(file_path)
    if "reconstruction_error" not in df.columns:
        print(f"[ERROR] Invalid anomaly result format in {file_path}")
        return

    df["severity_score"] = compute_severity(df["reconstruction_error"])
    df["estimated_RUL"] = np.nan
    df["health_state"] = "Healthy"

    rul = estimate_rul(df["reconstruction_error"], sampling_interval)
    latest_severity = df["severity_score"].iloc[-1]

    # classify health state
    if latest_severity < 0.3:
        state = "Healthy"
    elif latest_severity < 0.7:
        state = "Degrading"
    else:
        state = "Critical"

    df.loc[:, "estimated_RUL"] = rul
    df.loc[:, "health_state"] = state

    output_path = file_path.replace(".csv", "_prognostics.csv")
    df.to_csv(output_path, index=False)

    print(f"[INFO] Prognostics (severity + RUL) saved to {output_path}")
    print(f"[INFO] Severity Score: {latest_severity:.2f}")
    if rul == np.inf:
        print("[INFO] No degradation trend detected (RUL = ∞)")
    else:
        print(f"[INFO] Estimated RUL: {rul:.2f} time units")
    print(f"[INFO] Health State: {state}")

    return df
