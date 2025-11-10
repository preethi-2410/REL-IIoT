import pandas as pd
import matplotlib.pyplot as plt

# Load the log file
df = pd.read_csv("logs/adaptive_realtime_log.csv",
                 names=["timestamp", "motor", "error", "threshold", "status"])

motors = df["motor"].unique()

for motor in motors:
    data = df[df["motor"] == motor]
    plt.figure(figsize=(10, 5))
    plt.plot(data["error"], label="Reconstruction Error")
    plt.plot(data["threshold"], label="Adaptive Threshold", linestyle="--")
    plt.title(f"Motor {motor} - Anomaly Detection Trend")
    plt.xlabel("Time Steps")
    plt.ylabel("Error / Threshold")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
