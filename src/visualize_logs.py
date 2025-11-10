import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import re
from datetime import datetime
import os

log_file = "logs/M1_train_log.txt"

if not os.path.exists(log_file):
    print(f"Log file not found: {log_file}")
    exit()

timestamps, accs, f1s, faults = [], [], [], []

# Parse lines like: "Thu Nov 6 18:32:21 2025 | acc=0.940, f1=0.940, fault_rate=0.050"
with open(log_file, "r") as f:
    for line in f:
        match = re.search(
            r"(\w+\s+\w+\s+\d+\s+\d+:\d+:\d+\s+\d+).*?acc=(\d+\.\d+).*?f1=(\d+\.\d+).*?fault_rate=(\d+\.\d+)",
            line,
        )
        if match:
            ts, acc, f1, fault = match.groups()
            try:
                timestamps.append(datetime.strptime(ts, "%a %b %d %H:%M:%S %Y"))
                accs.append(float(acc))
                f1s.append(float(f1))
                faults.append(float(fault))
            except Exception:
                continue

if not timestamps:
    print("No valid entries found in the log file.")
    exit()

df = pd.DataFrame({"timestamp": timestamps, "accuracy": accs, "f1": f1s, "fault_rate": faults})

# --- Plot Accuracy and F1 ---
plt.figure(figsize=(10, 6))
plt.plot(df["timestamp"], df["accuracy"], marker="o", label="Accuracy", linewidth=2)
plt.plot(df["timestamp"], df["f1"], marker="x", label="F1 Score", linestyle="--", linewidth=2)
plt.title("Model Accuracy and F1 Score Over Time", fontsize=14)
plt.xlabel("Timestamp")
plt.ylabel("Metric Value")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.5)
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
plt.tight_layout()
plt.show()

# --- Plot Fault Rate ---
plt.figure(figsize=(10, 4))
plt.plot(df["timestamp"], df["fault_rate"], color="red", marker="s", linewidth=2)
plt.title("Fault Rate Over Time", fontsize=14)
plt.xlabel("Timestamp")
plt.ylabel("Fault Rate")
plt.grid(True, linestyle="--", alpha=0.5)
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
plt.tight_layout()
plt.show()
