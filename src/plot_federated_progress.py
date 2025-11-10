"""
plot_federated_progress.py
Plots the average reconstruction loss for each federated round
to visualize global model improvement across rounds.
"""

import os
import numpy as np
import matplotlib.pyplot as plt

RESULTS_DIR = "results"
SAVE_PATH = os.path.join(RESULTS_DIR, "federated_progress.png")

def load_losses():
    """Load average training losses from each motor per round."""
    rounds = {}
    for fname in os.listdir(RESULTS_DIR):
        if "_round" in fname and fname.endswith("_loss.npy"):
            path = os.path.join(RESULTS_DIR, fname)
            data = np.load(path)
            round_id = int(fname.split("_round")[1].split("_")[0])
            if round_id not in rounds:
                rounds[round_id] = []
            rounds[round_id].append(np.mean(data))
    return rounds

def plot_progress(rounds):
    """Plot the learning progress across federated rounds."""
    if not rounds:
        print("[ERROR] No federated loss data found in 'results' folder.")
        return

    round_nums = sorted(rounds.keys())
    avg_losses = [np.mean(rounds[r]) for r in round_nums]

    plt.figure(figsize=(8, 5))
    plt.plot(round_nums, avg_losses, marker="o", linewidth=2)
    plt.title("Federated Learning Progress", fontsize=14)
    plt.xlabel("Federated Round", fontsize=12)
    plt.ylabel("Average Reconstruction Loss", fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(SAVE_PATH)
    plt.show()

    print(f"[INFO] Plot saved to {SAVE_PATH}")

if __name__ == "__main__":
    rounds = load_losses()
    plot_progress(rounds)
