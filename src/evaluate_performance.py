"""
evaluate_performance.py
Evaluates anomaly detection and prognostic accuracy.
Includes SHAP interpretability for feature contribution analysis.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, roc_auc_score, confusion_matrix, roc_curve
from tensorflow.keras.models import load_model
from joblib import load
import shap

# -------------------------------------------------------------
MODEL_DIR = "models"
RESULTS_DIR = "results"
PLOTS_DIR = os.path.join(RESULTS_DIR, "plots")

MOTOR_ID = "M3_degrading"  # Change for other motors
MODEL_NAME = "global_autoencoder.h5"
SCALER_NAME = "global_scaler.pkl"
# -------------------------------------------------------------

os.makedirs(PLOTS_DIR, exist_ok=True)

# ---------- 1. Load data and model ----------
model_path = os.path.join(MODEL_DIR, MODEL_NAME)
scaler_path = os.path.join(MODEL_DIR, SCALER_NAME)
data_path = os.path.join(RESULTS_DIR, f"{MOTOR_ID}_anomaly_results.csv")
prog_path = os.path.join(RESULTS_DIR, f"{MOTOR_ID}_anomaly_results_prognostics.csv")

if not os.path.exists(data_path):
    raise FileNotFoundError(f"Results file not found: {data_path}")

print(f"[INFO] Evaluating model for {MOTOR_ID}")

scaler = load(scaler_path)
model = load_model(model_path, compile=False)

# Load data (from original sensor source if possible)
data_file = os.path.join("data", f"{MOTOR_ID}_sensor_data.csv")
df_data = pd.read_csv(data_file)
df_data = df_data.select_dtypes(include=[np.number])

if "fault" in df_data.columns:
    true_labels = df_data["fault"].values
    df_data = df_data.drop(columns=["fault"])
else:
    true_labels = np.zeros(len(df_data))

X_scaled = scaler.transform(df_data.values)

df_results = pd.read_csv(data_path)
pred_labels = df_results["anomaly"].values
recon_error = df_results["reconstruction_error"].values

# ---------- 2. Compute metrics ----------
precision = precision_score(true_labels, pred_labels, zero_division=0)
recall = recall_score(true_labels, pred_labels, zero_division=0)
f1 = f1_score(true_labels, pred_labels, zero_division=0)
acc = accuracy_score(true_labels, pred_labels)
roc_auc = roc_auc_score(true_labels, recon_error)

# ---------- 3. Confusion matrix ----------
cm = confusion_matrix(true_labels, pred_labels)

# ---------- 4. Plots ----------
plt.figure(figsize=(6, 5))
plt.imshow(cm, cmap="Blues")
plt.title(f"Confusion Matrix - {MOTOR_ID}")
plt.xlabel("Predicted")
plt.ylabel("True")
for i in range(2):
    for j in range(2):
        plt.text(j, i, cm[i, j], ha="center", va="center", color="black")
plt.savefig(os.path.join(PLOTS_DIR, f"{MOTOR_ID}_confusion_matrix.png"))
plt.close()

# Reconstruction error distribution
plt.figure(figsize=(8, 5))
plt.hist(recon_error[true_labels == 0], bins=50, alpha=0.7, label="Normal")
plt.hist(recon_error[true_labels == 1], bins=50, alpha=0.7, label="Fault")
plt.title("Reconstruction Error Distribution")
plt.legend()
plt.savefig(os.path.join(PLOTS_DIR, f"{MOTOR_ID}_error_distribution.png"))
plt.close()

# ROC curve
fpr, tpr, _ = roc_curve(true_labels, recon_error)
plt.figure()
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.savefig(os.path.join(PLOTS_DIR, f"{MOTOR_ID}_roc_curve.png"))
plt.close()

# RUL visualization if available
if os.path.exists(prog_path):
    df_prog = pd.read_csv(prog_path)
    if "estimated_RUL" in df_prog.columns:
        plt.figure(figsize=(10, 4))
        plt.plot(df_prog["estimated_RUL"], label="Predicted RUL")
        plt.title("Remaining Useful Life Trend")
        plt.xlabel("Sample Index")
        plt.ylabel("RUL (estimated)")
        plt.legend()
        plt.savefig(os.path.join(PLOTS_DIR, f"{MOTOR_ID}_rul_trend.png"))
        plt.close()

# ---------- 5. SHAP Analysis ----------
print("[INFO] Computing SHAP values (may take a few seconds)...")

# Take a small subset for speed
sample_X = X_scaled[:1000]
background = sample_X[np.random.choice(sample_X.shape[0], 100, replace=False)]

explainer = shap.KernelExplainer(model.predict, background)
shap_values = explainer.shap_values(sample_X[:100])

# Global summary plot
plt.figure()
shap.summary_plot(shap_values, sample_X[:100], feature_names=["vibration", "temperature", "current"], show=False)
plt.savefig(os.path.join(PLOTS_DIR, f"{MOTOR_ID}_shap_summary.png"))
plt.close()

# Local explanation (first anomaly)
anomaly_indices = np.where(pred_labels == 1)[0]
if len(anomaly_indices) > 0:
    i = anomaly_indices[0]
    shap.force_plot(explainer.expected_value[0], shap_values[0][i], sample_X[i],
                    feature_names=["vibration", "temperature", "current"],
                    matplotlib=True, show=False)
    plt.savefig(os.path.join(PLOTS_DIR, f"{MOTOR_ID}_shap_local.png"))
    plt.close()

# ---------- 6. Save evaluation summary ----------
report_path = os.path.join(RESULTS_DIR, f"{MOTOR_ID}_evaluation_report.txt")
with open(report_path, "w") as f:
    f.write(f"Model Evaluation Report for {MOTOR_ID}\n")
    f.write("="*60 + "\n\n")
    f.write(f"Accuracy:  {acc:.4f}\n")
    f.write(f"Precision: {precision:.4f}\n")
    f.write(f"Recall:    {recall:.4f}\n")
    f.write(f"F1-score:  {f1:.4f}\n")
    f.write(f"ROC-AUC:   {roc_auc:.4f}\n\n")
    f.write("Confusion Matrix:\n")
    f.write(str(cm) + "\n\n")
    f.write("Plots saved in: " + PLOTS_DIR + "\n")

print(f"\n[INFO] Evaluation complete. Report saved to {report_path}")
print("[INFO] SHAP summary and performance plots saved.")
