"""
train_motor.py
Trains Autoencoder models for healthy motors only (M1, M2, M4, M5).
Excludes 'fault' column to ensure only real sensor data is used.
"""

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from joblib import dump
from src.anomaly_model import build_autoencoder

# ---------------------------------------------------------------
DATA_DIR  = "data"
MODEL_DIR = "models"
LOG_DIR   = "logs"

HEALTHY_MOTORS = ["M1", "M2", "M4", "M5"]   # Exclude M3
EPOCHS = 50
BATCH_SIZE = 32
# ---------------------------------------------------------------

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

def train_motor(motor_id: str):
    print(f"\n--- Training Autoencoder for {motor_id} ---")
    data_path = os.path.join(DATA_DIR, f"{motor_id}_sensor_data.csv")

    if not os.path.exists(data_path):
        print(f"[ERROR] Missing data file for {motor_id}. Skipping.")
        return

    # Load and clean
    df = pd.read_csv(data_path)
    df = df.select_dtypes(include=[np.number])   # only numeric columns
    if "fault" in df.columns:
        df = df.drop(columns=["fault"])          # exclude fault label
    df = df.dropna()
    X = df.values.astype(float)

    # Normalize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Build model
    model = build_autoencoder(X_scaled.shape[1])
    history = model.fit(
        X_scaled, X_scaled,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_split=0.1,
        verbose=1
    )

    # Save artifacts
    model_path  = os.path.join(MODEL_DIR, f"{motor_id}_autoencoder.h5")
    scaler_path = os.path.join(MODEL_DIR, f"{motor_id}_scaler.pkl")
    loss_path   = os.path.join(LOG_DIR, f"{motor_id}_train_loss.npy")

    model.save(model_path, include_optimizer=False)
    dump(scaler, scaler_path)
    np.save(loss_path, history.history["loss"])

    print(f"[INFO] Model saved to {model_path}")
    print(f"[INFO] Scaler saved to {scaler_path}")

def main():
    for motor in HEALTHY_MOTORS:
        train_motor(motor)

if __name__ == "__main__":
    main()
