"""
federated_global_model.py
Combines healthy motor models (M1, M2, M4, M5) into a single global autoencoder
by averaging weights. This global model represents federated healthy behavior.
"""

import os
import numpy as np
from tensorflow.keras.models import load_model
from joblib import load, dump
from src.anomaly_model import build_autoencoder

# -------------------------------------------------------------
MODEL_DIR = "models"
GLOBAL_MODEL_PATH = os.path.join(MODEL_DIR, "global_autoencoder.h5")
GLOBAL_SCALER_PATH = os.path.join(MODEL_DIR, "global_scaler.pkl")

HEALTHY_MODELS = ["M1", "M2", "M4", "M5"]
# -------------------------------------------------------------

def average_weights(weight_list):
    """Compute element-wise average of multiple model weights."""
    new_weights = []
    for weights in zip(*weight_list):
        new_weights.append(np.mean(weights, axis=0))
    return new_weights

def federate_models():
    print("\n--- Federating Models: M1, M2, M4, M5 ---")

    model_paths = [os.path.join(MODEL_DIR, f"{m}_autoencoder.h5") for m in HEALTHY_MODELS]
    scaler_paths = [os.path.join(MODEL_DIR, f"{m}_scaler.pkl") for m in HEALTHY_MODELS]

    # Check all exist
    for path in model_paths + scaler_paths:
        if not os.path.exists(path):
            print(f"[ERROR] Missing: {path}")
            return

    # Load all models
    models = [load_model(p, compile=False) for p in model_paths]
    print(f"[INFO] Loaded {len(models)} healthy motor models.")

    # Check architecture consistency
    input_dim = models[0].input_shape[1]
    if not all(m.input_shape[1] == input_dim for m in models):
        print("[ERROR] Model input dimensions differ. Ensure consistent architectures.")
        return

    # Build new global autoencoder with same shape
    global_model = build_autoencoder(input_dim)

    # Average weights across all models
    weights_list = [m.get_weights() for m in models]
    averaged_weights = average_weights(weights_list)
    global_model.set_weights(averaged_weights)

    # Save global model
    global_model.save(GLOBAL_MODEL_PATH, include_optimizer=False)
    print(f"[INFO] Global federated model saved to {GLOBAL_MODEL_PATH}")

    # Average scalers (mean and scale)
    scalers = [load(p) for p in scaler_paths]
    means = np.mean([s.mean_ for s in scalers], axis=0)
    scales = np.mean([s.scale_ for s in scalers], axis=0)

    # Create new averaged scaler
    from sklearn.preprocessing import StandardScaler
    global_scaler = StandardScaler()
    global_scaler.mean_ = means
    global_scaler.scale_ = scales
    global_scaler.var_ = scales ** 2
    global_scaler.n_features_in_ = len(means)
    dump(global_scaler, GLOBAL_SCALER_PATH)

    print(f"[INFO] Global scaler saved to {GLOBAL_SCALER_PATH}")
    print("[SUCCESS] Federated global model creation complete.")

if __name__ == "__main__":
    federate_models()
