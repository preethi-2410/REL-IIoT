"""
federated_round_simulation.py
Simulates a complete federated learning round:
1. Distribute global model to all healthy edge devices
2. Locally retrain with new data
3. Aggregate weights to form a new global model
"""

import os
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler
from joblib import dump, load
from src.anomaly_model import build_autoencoder

# -------------------------------------------------------------
DATA_DIR = "data"
MODEL_DIR = "models"
LOG_DIR = "logs"

HEALTHY_MOTORS = ["M1", "M2", "M4", "M5"]
EPOCHS = 5          # few local epochs to simulate quick update
BATCH_SIZE = 32
# -------------------------------------------------------------

def average_weights(weight_list):
    """Compute element-wise average of multiple model weights."""
    new_weights = []
    for weights in zip(*weight_list):
        new_weights.append(np.mean(weights, axis=0))
    return new_weights

def run_federated_round():
    print("\n--- Starting Federated Learning Round ---")

    global_model_path = os.path.join(MODEL_DIR, "global_autoencoder.h5")
    global_scaler_path = os.path.join(MODEL_DIR, "global_scaler.pkl")

    if not os.path.exists(global_model_path) or not os.path.exists(global_scaler_path):
        print("[ERROR] Global model or scaler missing. Run federated_global_model.py first.")
        return

    global_model = load_model(global_model_path, compile=False)
    global_scaler = load(global_scaler_path)
    input_dim = global_model.input_shape[1]

    updated_weights = []
    updated_scalers = []

    for motor in HEALTHY_MOTORS:
        print(f"\n--- Local update for {motor} ---")

        data_path = os.path.join(DATA_DIR, f"{motor}_sensor_data.csv")
        if not os.path.exists(data_path):
            print(f"[WARNING] Missing data for {motor}. Skipping.")
            continue

        df = pd.read_csv(data_path)
        df = df.select_dtypes(include=[np.number]).dropna()
        # Use only the numeric feature columns used in training
        X = df.drop(columns=["fault"], errors="ignore").values


        # Scale using global scaler
        X_scaled = global_scaler.transform(X)

        # Clone the global model
        local_model = build_autoencoder(input_dim)
        local_model.set_weights(global_model.get_weights())

        # Train locally for a few epochs
        local_model.fit(X_scaled, X_scaled, epochs=EPOCHS,
                        batch_size=BATCH_SIZE, verbose=1)

        updated_weights.append(local_model.get_weights())

        # Update local scaler (for realism)
        scaler_local = StandardScaler()
        scaler_local.fit(X)
        updated_scalers.append(scaler_local)

    if not updated_weights:
        print("[ERROR] No local updates found.")
        return

    # Federated aggregation
    new_weights = average_weights(updated_weights)
    global_model.set_weights(new_weights)
    global_model.save(os.path.join(MODEL_DIR, "global_autoencoder_round2.h5"))
    print("[INFO] New global model saved as round 2.")

    # Average new scalers
    means = np.mean([s.mean_ for s in updated_scalers], axis=0)
    scales = np.mean([s.scale_ for s in updated_scalers], axis=0)
    global_scaler.mean_ = means
    global_scaler.scale_ = scales
    dump(global_scaler, os.path.join(MODEL_DIR, "global_scaler_round2.pkl"))
    print("[INFO] Global scaler updated and saved.")

    print("[SUCCESS] Federated Learning Round Complete.")

if __name__ == "__main__":
    run_federated_round()
