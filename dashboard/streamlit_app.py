import streamlit as st
import pandas as pd
import time
import numpy as np
from sensors.sensor_simulator import SensorSimulator

# ---------- PAGE SETUP ----------
st.set_page_config(page_title="Edge IIoT Anomaly Monitor", layout="wide")
st.title("🧠 Intelligent Anomaly Detection Dashboard")
st.markdown(
    "Real-time monitoring of **vibration**, **temperature**, and **current** "
    "with AI-based anomaly and health indicators."
)

# ---------- LAYOUT ----------
col1, col2 = st.columns([3, 1])
sim = SensorSimulator(machine_id="M1", interval=0.5)

# ---------- INITIAL PLACEHOLDERS ----------
data = []
anomaly_scores = []
fault_detected = False

sensor_chart = col1.empty()
anomaly_chart = col1.empty()
health_metric = col2.empty()
status_box = col2.empty()

# ---------- MAIN LOOP ----------
for i, r in enumerate(sim.stream()):
    data.append(r)
    df = pd.DataFrame(data)

    # Calculate health and anomaly scores
    vibration = df["vibration"].iloc[-1]
    temperature = df["temperature"].iloc[-1]
    health_score = np.clip(100 - (vibration * 5 + temperature / 2), 0, 100)
    anomaly_score = np.clip(1 - (health_score / 100), 0, 1)
    anomaly_scores.append(anomaly_score)

    # Update main charts
    with sensor_chart.container():
        st.subheader("📊 Live Sensor Data")
        st.line_chart(df[["vibration", "temperature", "current"]])

    with anomaly_chart.container():
        st.subheader("📈 Anomaly Score")
        st.line_chart(pd.DataFrame({"Anomaly Score": anomaly_scores}))

    # Update right panel
    health_metric.metric(label="Health Score", value=f"{health_score:.1f} %")

    if r["fault_label"] == 1 or anomaly_score > 0.4:
        status_box.error("🚨 Fault Detected!", icon="⚠️")
        fault_detected = True
    else:
        status_box.info("✅ Normal Operation")

    time.sleep(0.5)
    if i > 50:
        break

# ---------- END STATUS ----------
if fault_detected:
    st.warning("Simulation complete with faults detected. Investigate affected parameters.")
else:
    st.success("Simulation complete. All systems stable.")
