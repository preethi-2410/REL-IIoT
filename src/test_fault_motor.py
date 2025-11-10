"""
test_fault_motor.py
Evaluates fault motor (M3) using a healthy model (e.g., M1).
Compares reconstruction errors and visualizes anomalies.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from joblib import load

# ---------------------------------------------------------------
DATA_DIR = "data"
MODEL_DIR = "models"
RESULTS_DIR = "results"

TEST_MOTOR = "M3"     # Faulty motor
REF_MOTOR  = "global"     # Reference 
# ---------------------------------------------------------------

os.makedirs(RESULTS_DIR, exist_ok=True)

def compute_threshold(errors):
    """Adaptive threshold based on percentile."""
    return np.percentile(errors, 99.0)

def evaluate_fault_motor():
    print(f"\n--- Evaluating {TEST_MOTOR} using {REF_MOTOR}'s model ---")

    model_path = os.path.join(MODEL_DIR, f"{REF_MOTOR}_autoencoder.h5")
    scaler_path = os.path.join(MODEL_DIR, f"{REF_MOTOR}_scaler.pkl")
    data_path = os.path.join(DATA_DIR, f"{TEST_MOTOR}_sensor_data.csv")

    if not all(os.path.exists(p) for p in [model_path, scaler_path, data_path]):
        print("[ERROR] Missing required files. Check paths.")
        return

    model = load_model(model_path, compile=False)
    scaler = load(scaler_path)
    df = pd.read_csv(data_path)
    df = df.select_dtypes(include=[np.number])
    if "fault" in df.columns:
        df = df.drop(columns=["fault"])

    X = scaler.transform(df.values)

    reconstructed = model.predict(X)
    mse = np.mean(np.square(X - reconstructed), axis=1)

    threshold = compute_threshold(mse)
    anomalies = mse > threshold
    num_anomalies = np.sum(anomalies)

    print(f"[INFO] Threshold = {threshold:.6f}")
    print(f"[INFO] Anomalies detected = {num_anomalies} / {len(mse)}")

    results_path = os.path.join(RESULTS_DIR, f"{TEST_MOTOR}_fault_eval.csv")
    pd.DataFrame({
        "reconstruction_error": mse,
        "anomaly": anomalies.astype(int)
    }).to_csv(results_path, index=False)

    plt.figure(figsize=(10,4))
    plt.plot(mse, label="Reconstruction Error", linewidth=1)
    plt.axhline(y=threshold, color='r', linestyle='--', label=f"Threshold ({threshold:.4f})")
    plt.title(f"{TEST_MOTOR} evaluated with {REF_MOTOR}'s model")
    plt.xlabel("Sample Index")
    plt.ylabel("Reconstruction Error")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, f"{TEST_MOTOR}_fault_plot.png"))
    plt.close()

    print(f"[INFO] Results saved to {results_path}")

if __name__ == "__main__":
    evaluate_fault_motor()
