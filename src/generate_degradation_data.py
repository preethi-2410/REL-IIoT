import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import random

# Simulation parameters
num_points = 21600  # same length as your other datasets
start_time = datetime(2025, 11, 7, 2, 30, 0)
time_interval = timedelta(seconds=2)

# Base healthy operating ranges
base_vib = 1.0
base_temp = 65.0
base_curr = 10.5

# Gradual degradation (linear drift)
vib_drift = 0.0005      # vibration increases steadily
temp_drift = 0.0025     # temperature slowly rises
curr_drift = 0.0006     # current imbalance

# Generate time series
timestamps = [start_time + i*time_interval for i in range(num_points)]

vibration = [base_vib + vib_drift*i + np.random.normal(0, 0.05) for i in range(num_points)]
temperature = [base_temp + temp_drift*i + np.random.normal(0, 0.2) for i in range(num_points)]
current = [base_curr + curr_drift*np.sin(i/1000) + np.random.normal(0, 0.05) for i in range(num_points)]

# Gradual fault labeling (progressive failure)
fault = [0 if i < num_points*0.7 else 1 for i in range(num_points)]

# Create dataframe
df = pd.DataFrame({
    "timestamp": timestamps,
    "vibration": vibration,
    "temperature": temperature,
    "current": current,
    "fault": fault
})

# Save to data directory
df.to_csv("data/M3_degrading_sensor_data.csv", index=False)
print("✅ Degradation test data generated: data/M3_degrading_sensor_data.csv")
