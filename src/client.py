# src/client.py
import flwr as fl
import numpy as np
import pandas as pd
import sys, os
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

motor_id = sys.argv[1] if len(sys.argv) > 1 else "M1"

DATA_PATH = f"data/{motor_id}_sensor_data.csv"
MODEL_PATH = f"models/{motor_id}_autoencoder.h5"
SCALER_PATH = f"models/{motor_id}_scaler.npy"
THRESH_PATH = f"models/{motor_id}_threshold.npy"

# ------------------------------------------------------------
# Load and clean data
# ------------------------------------------------------------
df = pd.read_csv(DATA_PATH)
feature_cols = ["vibration", "temperature", "current"]

# keep only normal samples
df_norm = df[df["fault"] == 0].reset_index(drop=True)
X = df_norm[feature_cols].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
np.save(SCALER_PATH, scaler.mean_)
np.save(SCALER_PATH.replace(".npy", "_scale.npy"), scaler.scale_)

input_dim = X_scaled.shape[1]
print(f"[{motor_id}] ✅ Dataset shape: {X_scaled.shape}, Fault rate: {df['fault'].mean():.3f}")

# ------------------------------------------------------------
# Build autoencoder
# ------------------------------------------------------------
def make_autoencoder(input_dim):
    model = Sequential([
        Dense(16, activation='relu', input_shape=(input_dim,)),
        Dense(8, activation='relu'),
        Dense(4, activation='relu'),
        Dense(8, activation='relu'),
        Dense(16, activation='relu'),
        Dense(input_dim, activation='linear')
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

model = make_autoencoder(input_dim)

# ------------------------------------------------------------
# Flower client definition
# ------------------------------------------------------------
class MotorClient(fl.client.NumPyClient):
    def get_parameters(self, config):
        return model.get_weights()

    def fit(self, parameters, config):
        model.set_weights(parameters)
        model.fit(X_scaled, X_scaled, epochs=30, batch_size=64, verbose=0)
        recons = model.predict(X_scaled, verbose=0)
        mse = np.mean(np.square(X_scaled - recons), axis=1)
        threshold = np.mean(mse) + 3*np.std(mse)
        np.save(THRESH_PATH, threshold)
        model.save(MODEL_PATH)
        print(f"[{motor_id}] 🧾 Train | loss={np.mean(mse):.6f} | threshold={threshold:.6f}")
        return model.get_weights(), len(X_scaled), {"loss": float(np.mean(mse))}

    def evaluate(self, parameters, config):
        model.set_weights(parameters)
        recons = model.predict(X_scaled, verbose=0)
        mse = np.mean(np.square(X_scaled - recons), axis=1)
        loss = float(np.mean(mse))
        acc = float(np.mean(mse < np.load(THRESH_PATH)))
        print(f"[{motor_id}] 🧠 Eval | acc={acc:.3f}, loss={loss:.6f}")
        return loss, len(X_scaled), {"accuracy": acc, "loss": loss}

# ------------------------------------------------------------
# Start client
# ------------------------------------------------------------
print(f"[{motor_id}] ⚙ Starting Flower client on localhost:8080...")
fl.client.start_numpy_client(server_address="localhost:8080", client=MotorClient())
