# src/server.py
import flwr as fl
import numpy as np

NUM_ROUNDS = 3
print("\n🌐 Persistent Federated Server started... waiting for clients.")
print("------------------------------------------------------------")

# ------------------------------------------------------------
# Custom robust aggregator (trimmed mean)
# ------------------------------------------------------------
def trimmed_mean(results, trim_ratio=0.2):
    # results is list of (parameters, num_examples, metrics)
    if not results:
        return None, {}
    weights = [r[0] for r in results]
    num_clients = len(weights)
    stacked = [np.array(w, dtype=object) for w in zip(*weights)]
    aggregated = []
    for layer_weights in stacked:
        layer_arr = np.stack(layer_weights)
        k = int(num_clients * trim_ratio)
        layer_arr.sort(axis=0)
        trimmed = layer_arr[k:num_clients - k] if num_clients > 2*k else layer_arr
        aggregated.append(np.mean(trimmed, axis=0))
    avg_metrics = {"mean_loss": np.mean([r[2]["loss"] for r in results if "loss" in r[2]])}
    return aggregated, avg_metrics

# ------------------------------------------------------------
# Strategy setup
# ------------------------------------------------------------
strategy = fl.server.strategy.FedAvg(
    fraction_fit=1.0,
    fraction_evaluate=1.0,
    min_fit_clients=1,
    min_evaluate_clients=1,
    min_available_clients=1,
    evaluate_metrics_aggregation_fn=lambda metrics: {
        "avg_loss": np.mean([m["loss"] for _, m in metrics if "loss" in m]),
        "avg_acc": np.mean([m["accuracy"] for _, m in metrics if "accuracy" in m]),
    },
)

# ------------------------------------------------------------
# Launch the server
# ------------------------------------------------------------
fl.server.start_server(
    server_address="localhost:8080",
    config=fl.server.ServerConfig(num_rounds=NUM_ROUNDS),
    strategy=strategy,
)
