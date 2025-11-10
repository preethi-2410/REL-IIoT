"""
federated_multi_rounds.py
Simulates multiple federated learning rounds where healthy edge models
(M1, M2, M4, M5) locally retrain and collaboratively update the global model.
"""

import os
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler
from joblib import load, dump
from src.anomaly_model import build_autoencoder

# ------------------------------------------------------------
DATA_DIR = "data"
MODEL_DIR = "models"
RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

HEALTHY_MOTORS = ["M1", "M2", "M4", "M5"]
EPOCHS = 5              # small local updates each round
BATCH_SIZE = 32
TOTAL_ROUNDS = 3        # number of federated rounds to simulate
# ------------------------------------------------------------

def average_weights(weight_list):
    """Compute element-wise average of multiple model weights."""
    return [np.mean(w, axis=0) for w in zip(*weight_list)]

def federated_round(round_id, global_model, global_scaler):
    """Simulate one federated learning round."""
    print(f"\n=== Federated Learning Round {round_id} ===")
    input_dim = global_model.input_shape[1]
    updated_weights, updated_scalers = [], []

    for motor in HEALTHY_MOTORS:
        print(f"\n--- Local update for {motor} ---")
        data_path = os.path.join(DATA_DIR, f"{motor}_sensor_data.csv")
        if not os.path.exists(data_path):
            print(f"[WARNING] Missing data for {motor}. Skipping.")
            continue

        df = pd.read_csv(data_path)
        df = df.select_dtypes(include=[np.number]).dropna()
        X = df.drop(columns=["fault"], errors="ignore").values

        # Scale using global scaler
        X_scaled = global_scaler.transform(X)

        # Clone and retrain
        local_model = build_autoencoder(input_dim)
        local_model.set_weights(global_model.get_weights())
        history = local_model.fit(
            X_scaled, X_scaled,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            verbose=1
        )

        updated_weights.append(local_model.get_weights())

        # Update local scaler for realism
        scaler_local = StandardScaler()
        scaler_local.fit(X)
        updated_scalers.append(scaler_local)

        # Save local training loss for reference
        np.save(os.path.join(RESULTS_DIR, f"{motor}_round{round_id}_loss.npy"),
                history.history["loss"])

    if not updated_weights:
        print("[ERROR] No local updates found.")
        return global_model, global_scaler

    # Aggregate weights
    new_weights = average_weights(updated_weights)
    global_model.set_weights(new_weights)

    # Aggregate scalers
    means = np.mean([s.mean_ for s in updated_scalers], axis=0)
    scales = np.mean([s.scale_ for s in updated_scalers], axis=0)
    global_scaler.mean_ = means
    global_scaler.scale_ = scales
    dump(global_scaler, os.path.join(MODEL_DIR, f"global_scaler_round{round_id}.pkl"))

    # Save model
    new_model_path = os.path.join(MODEL_DIR, f"global_autoencoder_round{round_id}.h5")
    global_model.save(new_model_path)
    print(f"[INFO] Saved global model for Round {round_id} to {new_model_path}")

    return global_model, global_scaler

def simulate_federated_learning():
    print("\n--- Starting Federated Multi-Round Simulation ---")

    model_path = os.path.join(MODEL_DIR, "global_autoencoder.h5")
    scaler_path = os.path.join(MODEL_DIR, "global_scaler.pkl")
    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        print("[ERROR] Missing base global model or scaler.")
        return

    global_model = load_model(model_path, compile=False)
    global_scaler = load(scaler_path)

    for round_id in range(2, TOTAL_ROUNDS + 2):  # e.g. Round2–Round4
        global_model, global_scaler = federated_round(round_id, global_model, global_scaler)

    print("\n[SUCCESS] All federated rounds completed successfully!")

if __name__ == "__main__":
    simulate_federated_learning()
