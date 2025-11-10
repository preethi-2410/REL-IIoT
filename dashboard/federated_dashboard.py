import os
import time
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

LOG_DIR = "logs"
DATA_DIR = "data"

st.set_page_config(page_title="Federated IIoT Dashboard", layout="wide")

st.title("🏭 Federated IIoT Training Dashboard")
st.markdown("Monitoring distributed model training and machine health across edge devices.")

# Sidebar refresh interval
interval = st.sidebar.slider("Refresh Interval (seconds)", 5, 60, 10)
st.sidebar.markdown("🕒 Auto-refresh every {} seconds".format(interval))

# Helper to parse federated training logs
def parse_logs():
    data = []
    for file in os.listdir(LOG_DIR):
        if file.endswith("_train_log.txt"):
            machine = file.split("_")[0]
            with open(os.path.join(LOG_DIR, file), "r") as f:
                for line in f:
                    if "Local Training Completed" in line:
                        parts = line.strip().split("|")
                        if len(parts) >= 3:
                            timestamp = parts[0].strip()
                            acc = float(parts[2].split("=")[1].split(",")[0])
                            f1 = float(parts[2].split("f1=")[1])
                            data.append([machine, timestamp, acc, f1])
    if data:
        df = pd.DataFrame(data, columns=["Machine", "Timestamp", "Accuracy", "F1"])
        df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")
        df = df.sort_values(by="Timestamp")
        return df
    else:
        return pd.DataFrame(columns=["Machine", "Timestamp", "Accuracy", "F1"])


# Helper to load fault data
def parse_fault_data():
    all_faults = []
    for file in os.listdir(DATA_DIR):
        if file.endswith(".csv"):
            try:
                df = pd.read_csv(os.path.join(DATA_DIR, file))
                machine = file.replace(".csv", "")
                if "fault" in df.columns:
                    fault_counts = df["fault"].value_counts().reset_index()
                    fault_counts.columns = ["Fault Type", "Count"]
                    fault_counts["Machine"] = machine
                    all_faults.append(fault_counts)
            except Exception as e:
                st.warning(f"Error reading {file}: {e}")
    if all_faults:
        return pd.concat(all_faults, ignore_index=True)
    else:
        return pd.DataFrame(columns=["Fault Type", "Count", "Machine"])


# Main section
placeholder = st.empty()

while True:
    df_logs = parse_logs()
    df_faults = parse_fault_data()

    with placeholder.container():
        st.subheader("📊 Latest Federated Training Summary")
        if df_logs.empty:
            st.warning("No training logs found yet.")
        else:
            latest = df_logs.groupby("Machine").last().reset_index()
            st.dataframe(latest[["Machine", "Accuracy", "F1", "Timestamp"]])

            col1, col2 = st.columns(2)
            with col1:
                fig1, ax1 = plt.subplots()
                for m in df_logs["Machine"].unique():
                    sub = df_logs[df_logs["Machine"] == m]
                    ax1.plot(sub["Timestamp"], sub["Accuracy"], label=m)
                ax1.set_title("Accuracy Over Time")
                ax1.legend()
                st.pyplot(fig1)

            with col2:
                fig2, ax2 = plt.subplots()
                for m in df_logs["Machine"].unique():
                    sub = df_logs[df_logs["Machine"] == m]
                    ax2.plot(sub["Timestamp"], sub["F1"], label=m)
                ax2.set_title("F1 Score Over Time")
                ax2.legend()
                st.pyplot(fig2)

        st.markdown("---")
        st.subheader("⚙️ Fault Distribution per Machine")

        if df_faults.empty:
            st.warning("No fault data yet. Run your sensor simulator.")
        else:
            machines = df_faults["Machine"].unique()
            cols = st.columns(len(machines))
            for i, m in enumerate(machines):
                sub = df_faults[df_faults["Machine"] == m]
                fig, ax = plt.subplots()
                ax.pie(sub["Count"], labels=sub["Fault Type"], autopct="%1.1f%%", startangle=90)
                ax.set_title(m)
                cols[i].pyplot(fig)

    time.sleep(interval)
