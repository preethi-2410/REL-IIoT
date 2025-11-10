import time
import random
import csv
from datetime import datetime
import numpy as np
import os


class SensorSimulator:
    """
    Multi-fault industrial sensor simulator.
    Simulates vibration, temperature, and current data for multiple machine types
    and injects realistic fault patterns correlated with physical behavior.
    """

    def __init__(self, machine_id="M1", machine_type="motor", interval=0.5, duration=60):
        self.machine_id = machine_id
        self.machine_type = machine_type.lower()
        self.interval = interval
        self.duration = duration

        # Machine-specific base parameters
        self.base_params = {
            "motor": {"vib": (5, 0.8), "temp": (60, 2), "cur": (2, 0.2)},
            "pump": {"vib": (3, 0.5), "temp": (55, 1.5), "cur": (1.5, 0.15)},
            "compressor": {"vib": (7, 1), "temp": (70, 3), "cur": (3, 0.3)},
            "generator": {"vib": (4, 0.6), "temp": (50, 2), "cur": (2.5, 0.25)},
        }

        self.output_file = f"data/{self.machine_id}_{self.machine_type}_data.csv"
        os.makedirs("data", exist_ok=True)

        # Fault configuration probabilities
        self.fault_types = {
            0: "Normal",
            1: "Bearing Wear",
            2: "Overheating",
            3: "Misalignment",
            4: "Electrical Surge",
        }

    def inject_fault(self, base_vib, base_temp, base_cur, fault_type):
        """Inject realistic sensor disturbances based on fault type"""
        if fault_type == 1:  # Bearing wear
            base_vib += np.random.uniform(2.0, 4.0)
            base_temp += np.random.uniform(0.2, 0.5)
        elif fault_type == 2:  # Overheating
            base_temp += np.random.uniform(10, 20)
            base_vib += np.random.uniform(0.5, 1.0)
        elif fault_type == 3:  # Misalignment
            base_vib += np.random.uniform(-3, 6) * np.sin(time.time())
            base_cur += np.random.uniform(0.1, 0.3)
        elif fault_type == 4:  # Electrical surge
            base_cur += np.random.uniform(2, 4)
            base_temp += np.random.uniform(3, 5)
        return base_vib, base_temp, base_cur

    def stream(self):
        """Continuous generator yielding simulated readings"""
        start = time.time()
        while (time.time() - start) < self.duration:
            params = self.base_params.get(self.machine_type, self.base_params["motor"])
            vib = np.random.normal(*params["vib"])
            temp = np.random.normal(*params["temp"])
            cur = np.random.normal(*params["cur"])

            # Randomly inject a fault
            fault_type = np.random.choice([0, 0, 0, 1, 2, 3, 4], p=[0.7, 0.1, 0.1, 0.05, 0.03, 0.01, 0.01])
            if fault_type != 0:
                vib, temp, cur = self.inject_fault(vib, temp, cur, fault_type)

            reading = {
                "timestamp": datetime.utcnow().isoformat(),
                "vibration": round(vib, 2),
                "temperature": round(temp, 2),
                "current": round(cur, 2),
                "fault_label": fault_type,
                "fault_type": self.fault_types[fault_type],
            }

            yield reading
            time.sleep(self.interval)

    def run(self, num_samples=200):
        """Run simulation and save to CSV"""
        print(f"[{self.machine_id}] 🧩 Simulating {self.machine_type.upper()} | Output: {self.output_file}")

        with open(self.output_file, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["timestamp", "vibration", "temperature", "current", "fault_label", "fault_type"])
            writer.writeheader()

            for i, r in enumerate(self.stream()):
                writer.writerow(r)
                print(f"{i:04d} | {r['timestamp']} | vib={r['vibration']} temp={r['temperature']} cur={r['current']} fault={r['fault_type']}")
                if i >= num_samples:
                    break

        print(f"\n[{self.machine_id}] ✅ Generated {num_samples} readings for {self.machine_type}.\n")


if __name__ == "__main__":
    available = ["motor", "pump", "compressor", "generator"]
    print(f"Available machine types: {', '.join(available)}")
    mtype = input("Enter machine type: ").strip().lower()
    mid = input("Enter machine ID (e.g., M1, M2, M3): ").strip()
    sim = SensorSimulator(machine_id=mid, machine_type=mtype, interval=0.5, duration=120)
    sim.run(num_samples=200)
