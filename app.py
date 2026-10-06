import streamlit as st
import joblib
import pandas as pd


# Load trained model and settings
artifact = joblib.load("smart_charger_model.pkl")

model = artifact["model"]
scaler = artifact["scaler"]
threshold = artifact["threshold"]
nominal_voltage = artifact["nominal_voltage"]
voltage_tolerance = artifact["voltage_tolerance"]


# Page configuration
st.set_page_config(
    page_title="Smart Charger Health Analyzer",
    page_icon="🔋",
    layout="centered"
)


# Store charger readings
if "readings" not in st.session_state:
    st.session_state.readings = []


# Title
st.title("🔋 Smart Charger Health Analyzer")

st.write(
    "AI-based system for analyzing charger voltage, current and power."
)


# Input section
st.subheader("🔌 Charger Input")

voltage = st.number_input(
    "Output Voltage (V)",
    min_value=0.0,
    max_value=20.0,
    value=5.0,
    step=0.1
)

current = st.number_input(
    "Output Current (A)",
    min_value=0.0,
    max_value=10.0,
    value=0.5,
    step=0.1
)


# Check Charger Health button
if st.button("🔍 Check Charger Health"):

    # Create input DataFrame
    new_data = pd.DataFrame({
        "Output Voltage RMS (V)": [voltage],
        "Output Current RMS (A)": [current],
        "Output Active Power (W)": [0]
    })


    # Scale the input data
    new_scaled = scaler.transform(new_data)


    # Use voltage and current as model inputs
    new_input = new_scaled[:, :2]


    # Predict power
    predicted_scaled_power = model.predict(new_input)


    # Convert predicted power back to original scale
    predicted_power = scaler.inverse_transform(
        pd.DataFrame(
            [[
                new_scaled[0, 0],
                new_scaled[0, 1],
                predicted_scaled_power[0]
            ]],
            columns=[
                "Output Voltage RMS (V)",
                "Output Current RMS (A)",
                "Output Active Power (W)"
            ]
        )
    )[0, 2]


    # Calculate actual power
    actual_power = voltage * current


    # Calculate prediction error
    error = abs(actual_power - predicted_power)


    # Check voltage range
    voltage_ok = (
        nominal_voltage - voltage_tolerance
        <= voltage
        <= nominal_voltage + voltage_tolerance
    )


    # Determine charger health
    if error > threshold and not voltage_ok:

        health_status = "ABNORMAL"

    elif error > threshold or not voltage_ok:

        health_status = "WARNING"

    else:

        health_status = "NORMAL"


    # Recommendation
    if health_status == "NORMAL":

        recommendation = (
            "Charger readings are within the expected pattern. "
            "Continue normal use."
        )

    elif health_status == "WARNING":

        recommendation = (
            "Check the charger for possible damage and avoid using it "
            "until the issue is inspected."
        )

    else:

        recommendation = (
            "Stop using the charger and inspect it for possible damage."
        )


    # Store reading for graphs
    reading_number = len(st.session_state.readings) + 1

    st.session_state.readings.append({
        "Reading": reading_number,
        "Voltage": voltage,
        "Current": current,
        "Power": actual_power
    })


    # -------------------------
    # Charger Measurements
    # -------------------------

    st.subheader("📊 Charger Measurements")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Voltage",
            f"{voltage:.3f} V"
        )

    with col2:
        st.metric(
            "Current",
            f"{current:.3f} A"
        )

    with col3:
        st.metric(
            "Power",
            f"{actual_power:.3f} W"
        )


    # -------------------------
    # AI Analysis
    # -------------------------

    st.subheader("🤖 AI Analysis")

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "Predicted Power",
            f"{predicted_power:.3f} W"
        )

    with col2:
        st.metric(
            "Prediction Error",
            f"{error:.3f} W"
        )


    # -------------------------
    # Charger Health
    # -------------------------

    st.subheader("🔋 Charger Health")

    if health_status == "NORMAL":

        st.success("🟢 Charger Health: NORMAL")

    elif health_status == "WARNING":

        st.warning("🟡 Charger Health: WARNING")

    else:

        st.error("🔴 Charger Health: ABNORMAL")


    # -------------------------
    # Recommendation
    # -------------------------

    st.subheader("💡 Recommendation")

    st.info(recommendation)


# -------------------------
# Graphs
# -------------------------

if len(st.session_state.readings) > 0:

    readings_df = pd.DataFrame(st.session_state.readings)


    # Voltage graph
    st.subheader("📈 Voltage Stability")

    voltage_chart = readings_df.set_index("Reading")[["Voltage"]]

    st.line_chart(voltage_chart)


    # Power graph
    st.subheader("⚡ Power Monitoring")

    power_chart = readings_df.set_index("Reading")[["Power"]]

    st.line_chart(power_chart)