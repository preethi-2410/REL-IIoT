"""
adaptive_recalibration.py
Adaptive threshold recalibration using live data.
Evaluates M3 using a healthy model and continuously updates threshold.
Integrates fault severity and Remaining Useful Life (RUL) estimation.
"""

import os
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from joblib import load
from src.prognostics import add_prognostics_to_results

# ---------------------------------------------------------------
DATA_DIR = "data"
MODEL_DIR = "models"
LOG_DIR = "logs"
RESULTS_DIR = "results"

TEST_MOTOR = "M3_degrading"
REFERENCE_MODEL = "global"   # model trained from federated aggregation

WINDOW_SIZE = 1000    # samples per recalibration window
ALPHA = 0.2           # smoothing factor for adaptive update
K_STD = 3.0           # threshold = mean + K_STD * std
# ---------------------------------------------------------------

os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)


def compute_reconstruction_error(model, X_scaled):
    """Reconstruct input using the model and return per-sample error."""
    reconstructed = model.predict(X_scaled, verbose=0)
    return np.mean(np.square(X_scaled - reconstructed), axis=1)


def adaptive_threshold_update(old_thresh, new_errors):
    """Update threshold adaptively using exponential smoothing."""
    mean_err, std_err = np.mean(new_errors), np.std(new_errors)
    adaptive_target = mean_err + K_STD * std_err
    new_thresh = (1 - ALPHA) * old_thresh + ALPHA * adaptive_target
    return new_thresh, mean_err, std_err


def main():
    model_path = os.path.join(MODEL_DIR, f"{REFERENCE_MODEL}_autoencoder.h5")
    scaler_path = os.path.join(MODEL_DIR, f"{REFERENCE_MODEL}_scaler.pkl")
    data_path = os.path.join(DATA_DIR, f"{TEST_MOTOR}_sensor_data.csv")

    if not all(os.path.exists(p) for p in [model_path, scaler_path, data_path]):
        print("[ERROR] Missing files. Check model/scaler/data paths.")
        return

    print(f"\n--- Adaptive Threshold Recalibration for {TEST_MOTOR} using {REFERENCE_MODEL}'s model ---")

    model = load_model(model_path, compile=False)
    scaler = load(scaler_path)
    df = pd.read_csv(data_path)

    # Keep only numeric features
    df = df.select_dtypes(include=[np.number])

    if "fault" in df.columns:
        faults = df["fault"].values
        df = df.drop(columns=["fault"])
    else:
        faults = np.zeros(len(df))

    X = scaler.transform(df.values)

    # Initial threshold from first window
    init_errors = compute_reconstruction_error(model, X[:WINDOW_SIZE])
    thresh = np.mean(init_errors) + K_STD * np.std(init_errors)
    print(f"[INIT] Starting threshold = {thresh:.4f}")

    log_path = os.path.join(LOG_DIR, "adaptive_realtime_log.csv")
    results_path = os.path.join(RESULTS_DIR, f"{TEST_MOTOR}_anomaly_results.csv")

    # Create CSV log headers
    with open(log_path, "w") as f:
        f.write("window_start,window_end,threshold,mean_error,std_error,anomalies,fault_rate\n")

    all_errors = []

    for start in range(0, len(X), WINDOW_SIZE):
        end = start + WINDOW_SIZE
        window = X[start:end]
        fault_window = faults[start:end]
        if len(window) == 0:
            break

        window_errors = compute_reconstruction_error(model, window)
        new_thresh, mean_err, std_err = adaptive_threshold_update(thresh, window_errors)
        anomalies = np.sum(window_errors > new_thresh)
        fault_rate = np.mean(fault_window)

        with open(log_path, "a") as f:
            f.write(f"{start},{end},{new_thresh:.6f},{mean_err:.6f},{std_err:.6f},{anomalies},{fault_rate:.3f}\n")

        print(f"[Window {start}-{end}] Threshold: {new_thresh:.4f}, Mean: {mean_err:.4f}, Fault%: {fault_rate*100:.1f}%")

        # Update threshold adaptively
        thresh = new_thresh
        all_errors.extend(window_errors)

    print(f"\n[INFO] Adaptive recalibration log saved to {log_path}")

    # Save final anomaly detection results
    df_results = pd.DataFrame({
        "reconstruction_error": all_errors,
        "anomaly": (np.array(all_errors) > thresh).astype(int)
    })
    df_results.to_csv(results_path, index=False)
    print(f"[INFO] Anomaly results saved to {results_path}")

    # --- Prognostics Integration ---
    print(f"[INFO] Running prognostics analysis for {TEST_MOTOR} ...")
    add_prognostics_to_results(results_path)
    print(f"[INFO] Prognostics (severity + RUL) added successfully.\n")


if __name__ == "__main__":
    main()
