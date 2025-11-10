"""
xai_explain.py
Lightweight explainability for Autoencoder-based anomaly detection.
Computes per-feature reconstruction errors and ranks top contributing features.
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)


def explain_sample(x_orig, x_recon, feature_names, top_k=2):
    """
    Compute feature-level reconstruction errors and identify top contributors.
    Returns both normalized importance and ranked features.
    """
    errors = np.abs(x_orig - x_recon)
    total_error = np.sum(errors)
    if total_error == 0:
        importance = np.zeros_like(errors)
    else:
        importance = errors / total_error

    sorted_idx = np.argsort(importance)[::-1]
    top_features = [(feature_names[i], float(importance[i])) for i in sorted_idx[:top_k]]

    return {
        "errors": errors.tolist(),
        "importance": importance.tolist(),
        "top_features": top_features
    }


def log_explanation(timestamp, motor_id, index, explanation):
    """
    Append human-readable feature explanations to CSV log.
    """
    log_path = os.path.join(RESULTS_DIR, f"{motor_id}_xai_log.csv")
    top_feats = ", ".join([f"{f} ({w:.2f})" for f, w in explanation["top_features"]])
    entry = pd.DataFrame([{
        "timestamp": timestamp,
        "sample_index": index,
        "top_features": top_feats
    }])
    entry.to_csv(log_path, mode="a", header=not os.path.exists(log_path), index=False)


def plot_feature_errors(feature_names, errors, motor_id, index):
    """
    Plot bar graph for feature-wise reconstruction errors of a single anomaly.
    """
    plt.figure(figsize=(4, 3))
    plt.bar(feature_names, errors, color="#FF8C00", alpha=0.9)
    plt.title(f"{motor_id} Feature Errors (Sample {index})")
    plt.ylabel("Error Magnitude")
    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, f"{motor_id}_xai_{index}.png")
    plt.savefig(path)
    plt.close()
