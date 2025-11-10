import numpy as np
import pandas as pd
import os
import random
from datetime import datetime, timedelta

def generate_motor_data(machine_id, hours=12, fault=False):
    """Simulate realistic 12-hour motor sensor data at 2-second intervals."""
    records = []
    timestamp = datetime.now()
    total_seconds = hours * 3600
    interval = 2  # every 2 seconds

    for t in range(0, total_seconds, interval):
        ts = timestamp + timedelta(seconds=t)

        # Normal behavior (healthy motor)
        vib = np.random.normal(0.4, 0.05)
        temp = np.random.normal(45, 1.5)
        cur = np.random.normal(8, 0.3)
        fault_label = 0

        # Faulty motor characteristics
        if fault:
            vib = np.random.normal(0.9, 0.1)
            temp = np.random.normal(65, 2.5)
            cur = np.random.normal(11.5, 0.5)
            if random.random() > 0.6:  # 40% of the time fault condition spikes
                fault_label = 1

        # Add a bit of drift over time (simulates wear)
        drift = t / total_seconds * 0.05
        vib += drift
        temp += drift * 10
        cur += drift * 2

        records.append([ts, vib, temp, cur, fault_label])

    df = pd.DataFrame(records, columns=["timestamp", "vibration", "temperature", "current", "fault"])
    df.to_csv(f"data/{machine_id}_sensor_data.csv", index=False)
    print(f"✅ {machine_id} data saved ({len(df)} records, fault={fault})")

def main():
    os.makedirs("data", exist_ok=True)
    machines = {
        "M1": False,
        "M2": False,
        "M3": True,   # Faulty motor
        "M4": False,
        "M5": False
    }

    for mid, is_faulty in machines.items():
        generate_motor_data(mid, hours=12, fault=is_faulty)

if __name__ == "__main__":
    main()
