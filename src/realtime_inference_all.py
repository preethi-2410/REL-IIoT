import os
import time
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import StandardScaler
import joblib
from datetime import datetime

# Paths
DATA_DIR = "data"
MODEL_DIR = "models"
LOG_DIR = "logs"

os.makedirs(LOG_DIR, exist_ok=True)

# List of motors
motors = ["M1", "M2", "M3", "M4", "M5"]

# Dictionaries to store models and scalers
models = {}
scalers = {}
thresholds = {}

print("\n🚀 Starting Real-Time Adaptive Edge Anomaly Monitor...\n")

# Load all models, scalers, and thresholds
for motor in motors:
    try:
        model_path = os.path.join(MODEL_DIR, f"{motor}_autoencoder.h5")
        threshold_path = os.path.join(MODEL_DIR, f"{motor}_threshold.npy")
        scaler_path = os.path.join(MODEL_DIR, f"{motor}_scaler.pkl")

        # Load model safely (ignore unrecognized metrics)
        model = keras.models.load_model(model_path, compile=False)
        print(f"[{motor}] ✅ Model loaded.")

        # Load threshold
        threshold = np.load(threshold_path)
        thresholds[motor] = float(threshold)
        print(f"[{motor}] ✅ Threshold loaded.")

        # Load scaler or fallback
        if os.path.exists(scaler_path):
            scaler = joblib.load(scaler_path)
            scalers[motor] = scaler
            print(f"[{motor}] ✅ Scaler loaded.")
        else:
            print(f"[{motor}] ⚠ Scaler not found — using auto normalization.")
            scalers[motor] = StandardScaler()

        models[motor] = model

    except Exception as e:
        print(f"[{motor}] ⚠ Error loading model or scaler: {e}")

# --- Fault Detection Mode for M3 ---
# Force M3 to use M1's healthy model and scaler
if "M1" in models:
    models["M3"] = models["M1"]
    scalers["M3"] = scalers["M1"]
    thresholds["M3"] = thresholds["M1"]
    print("\n[M3] ⚠ Using M1's healthy model for fault detection mode.\n")

print("------------------------------------------------------------")
print("Monitoring live sensor data for all available motors...")
print("------------------------------------------------------------\n")

# Prepare CSV logging
log_path = os.path.join(LOG_DIR, "adaptive_realtime_log.csv")
log_columns = ["timestamp", "motor", "error", "threshold", "status"]
log_df = pd.DataFrame(columns=log_columns)
log_df.to_csv(log_path, index=False)

# Real-time monitoring loop
try:
    while True:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"\n🕒 {timestamp}")
        print("------------------------------------------------------------")

        for motor in motors:
            try:
                data_path = os.path.join(DATA_DIR, f"{motor}_sensor_data.csv")
                data = pd.read_csv(data_path)

                data = pd.read_csv(data_path)

                # Drop non-numeric columns
                data = data.select_dtypes(include=[np.number])

                # Keep only the first 3 sensor columns (adjust if your training had a different count)
                if data.shape[1] > 3:
                    data = data.iloc[:, :3]

                if data.empty:
                    raise ValueError(f"No numeric sensor columns found in {motor}_sensor_data.csv")

                X = data.values

                # Scale data (fit if not already fitted)
                scaler = scalers[motor]
                if not hasattr(scaler, "mean_"):
                    scaler.fit(X)
                X_scaled = scaler.transform(X)

                # Run inference
                model = models[motor]
                reconstructions = model.predict(X_scaled, verbose=0)
                error = np.mean(np.square(X_scaled - reconstructions))
                threshold = thresholds[motor]

                status = "✅ NORMAL" if error <= threshold else "⚠ ANOMALY"
                print(f"[{motor}] Error={error:.6f} | Threshold={threshold:.6f} | Status={status}")

                # Log results
                new_row = pd.DataFrame([[timestamp, motor, error, threshold, status]], columns=log_columns)
                new_row.to_csv(log_path, mode="a", header=False, index=False)

            except Exception as e:
                print(f"[{motor}] ❌ Error during inference: {e}")

        time.sleep(8)  # Wait 8 seconds before next monitoring cycle

except KeyboardInterrupt:
    print("\n🛑 Monitoring stopped manually.")
    print(f"✅ Monitoring completed. Results and plots saved in '{LOG_DIR}/'.")
