"""
realtime_inference.py
Performs anomaly detection using trained autoencoders for each motor.
Computes adaptive thresholds, logs anomalies, and adds Explainable AI support.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from joblib import load
from src.xai_explain import explain_sample, log_explanation, plot_feature_errors

# -------------------------------------------------------------------
DATA_DIR = "data"
MODEL_DIR = "models"
RESULTS_DIR = "results"

MOTORS = ["M1", "M2", "M3_degrading", "M4", "M5"]
# -------------------------------------------------------------------

os.makedirs(RESULTS_DIR, exist_ok=True)


def compute_threshold(errors):
    """Compute adaptive threshold dynamically based on percentile."""
    return np.percentile(errors, 99.5)  # 99.5% of points are considered normal


def detect_anomalies(motor_id):
    print(f"\n--- Running anomaly detection for {motor_id} ---")

    model_path = os.path.join(MODEL_DIR, "global_autoencoder.h5")
    scaler_path = os.path.join(MODEL_DIR, "global_scaler.pkl")
    data_path = os.path.join(DATA_DIR, f"{motor_id}_sensor_data.csv")

    if not (os.path.exists(model_path) and os.path.exists(scaler_path) and os.path.exists(data_path)):
        print(f"[ERROR] Missing required files for {motor_id}. Skipping.")
        return None

    # Load model, scaler, and data
    model = load_model(model_path, compile=False)
    scaler = load(scaler_path)
    df = pd.read_csv(data_path)

    # --- Select only the correct 3 features (fix for mismatch) ---
    feature_cols = ["vibration", "temperature", "current"]
    df = df[feature_cols].copy()

    # Scale inputs
    X_scaled = scaler.transform(df.values)

    # Reconstruction
    reconstructed = model.predict(X_scaled, verbose=0)
    mse = np.mean(np.square(X_scaled - reconstructed), axis=1)

    # Adaptive threshold
    threshold = compute_threshold(mse)
    anomalies = mse > threshold

    # Fallback: ensure at least top 10 anomalies are explained
    if np.sum(anomalies) == 0:
        top_idx = np.argsort(mse)[-10:]  # top 10 highest reconstruction errors
    else:
        top_idx = np.where(anomalies)[0]

    # --- Explainability per anomaly sample ---
    feature_names = ["vibration", "temperature", "current"]

    for i in top_idx:
        explanation = explain_sample(X_scaled[i], reconstructed[i], feature_names)
        log_explanation(
            timestamp=df.iloc[i].get("timestamp") if "timestamp" in df.columns else i,
            motor_id=motor_id,
            index=i,
            explanation=explanation
        )
        plot_feature_errors(feature_names, explanation["errors"], motor_id, i)
    # --- End XAI ---

    # Log results
    results_path = os.path.join(RESULTS_DIR, f"{motor_id}_anomaly_results.csv")
    pd.DataFrame({
        "reconstruction_error": mse,
        "anomaly": anomalies.astype(int)
    }).to_csv(results_path, index=False)

    print(f"[INFO] Threshold for {motor_id}: {threshold:.6f}")
    print(f"[INFO] Anomalies detected: {np.sum(anomalies)} / {len(anomalies)}")
    print(f"[INFO] Results saved to {results_path}")

    # Plot reconstruction error
    plt.figure(figsize=(10, 4))
    plt.plot(mse, label="Reconstruction Error", linewidth=1)
    plt.axhline(y=threshold, color='r', linestyle='--', label=f"Threshold ({threshold:.4f})")
    plt.title(f"{motor_id} - Adaptive Threshold Anomaly Detection")
    plt.xlabel("Sample Index")
    plt.ylabel("Reconstruction Error")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, f"{motor_id}_anomaly_plot.png"))
    plt.close()

    return {
        "Motor": motor_id,
        "Threshold": threshold,
        "Anomalies": int(np.sum(anomalies)),
        "Total": len(anomalies)
    }


def main():
    summary = []
    for motor in MOTORS:
        result = detect_anomalies(motor)
        if result:
            summary.append(result)

    # Save summary CSV
    summary_path = os.path.join(RESULTS_DIR, "summary.csv")
    pd.DataFrame(summary).to_csv(summary_path, index=False)
    print(f"\n[INFO] Summary saved to {summary_path}")


if __name__ == "__main__":
    main()
