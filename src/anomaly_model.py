"""
anomaly_model.py
Defines the Autoencoder model architecture used for unsupervised anomaly detection.
"""

from tensorflow.keras import models, layers

def build_autoencoder(input_dim: int):
    """Builds and compiles a simple dense Autoencoder."""
    model = models.Sequential([
        layers.Dense(64, activation='relu', input_shape=(input_dim,)),
        layers.Dense(32, activation='relu'),
        layers.Dense(16, activation='relu'),
        layers.Dense(32, activation='relu'),
        layers.Dense(64, activation='relu'),
        layers.Dense(input_dim, activation='sigmoid')
    ])

    model.compile(optimizer='adam', loss='mse')
    return model
