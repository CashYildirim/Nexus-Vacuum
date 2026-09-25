

from collections import deque
import time
import numpy as np
import pandas as pd
import streamlit as st
import joblib

import importlib.util
from pathlib import Path

# --- Resolve all paths relative to this script's directory ---
HERE = Path(__file__).resolve().parent
_CANDIDATES = ["01_create_data.py", "veriuretimi.py", "veri_uretimi.py"]
GENERATOR_PATH = next((HERE / a for a in _CANDIDATES if (HERE / a).exists()), None)
MODEL_PATH = HERE / "nexus_model.pkl"
COLUMN_PATH = HERE / "model_columns.pkl"

if GENERATOR_PATH is None:
    st.error(
        f"Data generator file not found. Searched in: {HERE}\n\n"
        f"Searched names: {_CANDIDATES}\n\n"
        "If you changed the generator file name, add it to _CANDIDATES "
        "or copy it to the SAME directory as this script."
    )
    st.stop()

spec = importlib.util.spec_from_file_location("gen", GENERATOR_PATH)
gen = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gen)

SIGMA_SB = gen.SIGMA_SB
CFG = gen.CFG
CLASSES = ["NORMAL", "THERMAL OVERHEATING", "STEP LOSS", "VOLTAGE DROP"]

WINDOW_SIZE = 500       # Sliding window size
CONFIDENCE_THRESHOLD = 0.80
REFRESH_RATE_SEC = 1.0  # Refresh delay for display stability

st.set_page_config(page_title="NEXUS-VAKUM", page_icon="*", layout="wide")
st.title("NEXUS-VAKUM: Autonomous Space Manufacturing Robot")
st.caption("Visual telemetry and predictive maintenance panel "
           f"| sliding window N={WINDOW_SIZE} | confidence gate {CONFIDENCE_THRESHOLD}")


# Default feature order (Used if model_sutunlari.pkl cannot be read)
DEFAULT_COLUMNS = [
    "Ambient_Temperature_C", "Vacuum_Pressure_Torr", "Operating_Voltage_V",
    "Phase_Current_A", "Phase_Resistance_Ohm", "Motor_Speed_PPS",
    "Calculated_Temperature_C", "Torque_Stress_Coefficient",
]


@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"{MODEL_PATH} not found. Please run scripts in order:\n"
            "  python 01_create_data.py\n"
            "  python 02_baseline.py"
        )
    m = joblib.load(MODEL_PATH)
    cols = joblib.load(COLUMN_PATH) if COLUMN_PATH.exists() else list(DEFAULT_COLUMNS)
    return m, list(cols)


model, columns = None, list(DEFAULT_COLUMNS)
try:
    model, columns = load_model()
except Exception as err:
    st.error(f"Failed to load model.\n\n{type(err).__name__}: {err}")
    st.stop()

if model is None:
    st.error("Model could not be loaded.")
    st.stop()

# Check feature column alignment
_missing = set(columns) - set(DEFAULT_COLUMNS)
if _missing:
    st.error(
        "Saved model requires features not produced by the generator: "
        f"{sorted(_missing)}\n\n"
        "Likely cause: nexus_model.pkl was trained with older code. "
        "Delete it and re-run 02_baseline.py."
    )
    st.stop()


def generate_single_sample(rng):
    """Generates a single telemetry sample aligned with 01_create_data.py physics."""
    cfg = CFG
    prof = rng.choice(4, p=cfg["profile_weights"])
    U = lambda a, b: rng.uniform(a, b)

    par = [
        # ambient,   voltage,     current,     resistance, pps
        ((-50, 30), (6.8, 9.5), (.20, .34), (9, 16), (100, 1400),
         (.10, .45), (2, 6), (80, 220), (600, 3600)),
        ((20, 80), (6.5, 9.5), (.34, .45), (15, 20), (300, 1800),
         (.70, 1.0), (3, 4), (200, 450), (2400, 10800)),
        ((-20, 50), (6.0, 9.0), (.28, .42), (11, 19), (2000, 3200),
         (.45, .95), (6, 2), (100, 300), (1200, 7200)),
        ((-40, 40), (5.0, 6.3), (.34, .45), (9, 20), (200, 2000),
         (.20, .70), (2, 5), (80, 300), (600, 5400)),
    ][prof]

    ambient = U(*par[0]); voltage = U(*par[1]); current = U(*par[2])
    resistance = U(*par[3]); pps = U(*par[4])
    print_density = U(*par[5])
    lubricant = rng.beta(*par[6])
    R_conduction = U(*par[7]); duration = U(*par[8])
    pressure = 10 ** rng.uniform(*np.log10(cfg["pressure_Torr"]))

    # --- Transient thermal integration ---
    mc = cfg["mass_kg"] * cfg["specific_heat_J_kgK"]
    A = cfg["surface_area_m2"]
    epsF = cfg["emissivity"] * cfg["view_factor"]
    dt = cfg["time_step_s"]
    h = cfg["h_reference_W_m2K"] * pressure / (pressure + cfg["pressure_half_Torr"])
    Q = (current ** 2) * resistance * print_density
    T0 = ambient + 273.15
    T = T0
    for k in range(int(duration / dt)):
        T += dt * (Q - epsF * SIGMA_SB * A * (T ** 4 - cfg["space_temperature_K"] ** 4)
                   - h * A * (T - T0) - (T - T0) / R_conduction) / mc
    C = T - 273.15

    # --- Torque ---
    der = np.clip(1 - cfg["torque_derating_1_K"] * max(C - 25, 0), 0.25, 1.0)
    vf = np.clip(voltage / cfg["nominal_voltage_V"], 0.4, 1.0)
    pullout = (cfg["holding_torque_mNm"] * der * vf
               / np.sqrt(1 + (pps / cfg["corner_frequency_PPS"]) ** 2))
    friction = (cfg["friction_base_mNm"] + cfg["friction_gain_mNm"]
                * lubricant * (1 + 0.004 * max(C - 25, 0)))
    ts = (cfg["static_load_mNm"] + friction + 5 * print_density) / max(pullout, 1e-3)

    # --- Probabilistic labeling ---
    p_t = (1 / (1 + np.exp(-(C - cfg["T_sigmoid_center_C"]) / cfg["T_sigmoid_slope"]))
           if C > cfg["T_door_C"] else 0.0)
    p_s = (1 / (1 + np.exp(-(ts - cfg["S_sigmoid_center"]) / cfg["S_sigmoid_slope"]))
           if ts > cfg["S_door"] else 0.0)
    p_v = (cfg["p_voltage"] if (voltage < cfg["V_critical_V"]
                               and current > cfg["I_critical_A"]) else 0.0)
    P = np.array([max(1 - max(p_t, p_s, p_v), 0.0), p_t, p_s, p_v])
    P = P / P.sum()
    label = int(rng.choice(4, p=P))
    if rng.random() < cfg["label_noise"]:
        label = int(rng.integers(0, 4))

    row = {
        "Ambient_Temperature_C": ambient, "Vacuum_Pressure_Torr": pressure,
        "Operating_Voltage_V": voltage, "Phase_Current_A": current,
        "Phase_Resistance_Ohm": resistance, "Motor_Speed_PPS": pps,
        "Calculated_Temperature_C": C + rng.normal(0, cfg["thermistor_sigma_C"]),
        "Torque_Stress_Coefficient": ts + rng.normal(0, cfg["torque_sigma"]),
    }
    return row, label


# ---------------- Session State: Bounded History ----------------
if "window" not in st.session_state:
    st.session_state.window = deque(maxlen=WINDOW_SIZE)   # (is_correct, confidence)
    st.session_state.curve = deque(maxlen=200)
    st.session_state.rng = np.random.default_rng()
    st.session_state.counter = 0
    st.session_state.suspended = 0

is_live = st.sidebar.checkbox("Live stream simulation", value=True)
st.sidebar.markdown("---")
info_box = st.sidebar.empty()

c1, c2, c3, c4 = st.columns(4)
g1, g2 = st.columns(2)

if is_live:
    row_data, actual_label = generate_single_sample(st.session_state.rng)
    X = pd.DataFrame([row_data])[columns]
    prediction = int(model.predict(X)[0])
    confidence = float(model.predict_proba(X).max())

    st.session_state.window.append((prediction == actual_label, confidence))
    st.session_state.counter += 1
    if confidence < CONFIDENCE_THRESHOLD:
        st.session_state.suspended += 1

    c1.metric("Temperature", f"{row_data['Calculated_Temperature_C']:.1f} C")
    c2.metric("Voltage", f"{row_data['Operating_Voltage_V']:.2f} V")
    c3.metric("Speed", f"{int(row_data['Motor_Speed_PPS'])} PPS")
    c4.metric("Torque Stress", f"{row_data['Torque_Stress_Coefficient']:.2f}")

    if confidence < CONFIDENCE_THRESHOLD:
        st.warning(f"HOLD - autonomous action suspended "
                   f"(confidence {confidence*100:.1f}% < {CONFIDENCE_THRESHOLD*100:.0f}%) "
                   f"| provisional: {CLASSES[prediction]}")
    else:
        st.success(f"{CLASSES[prediction]}  |  confidence {confidence*100:.1f}%")

    # ---- Real Sliding Window ----
    correct_array = np.array([is_corr for is_corr, _ in st.session_state.window])
    n_samples = len(correct_array)
    if n_samples >= 30:
        accuracy = correct_array.mean()
        sigma = np.sqrt(accuracy * (1 - accuracy) / n_samples)
        st.session_state.curve.append(accuracy)
        info_box.info(
            f"**Sliding-window accuracy (N={n_samples})**\n\n"
            f"* Point estimate: %{accuracy*100:.2f}\n"
            f"* 95% CI: [%{(accuracy-1.96*sigma)*100:.2f}, "
            f"%{(accuracy+1.96*sigma)*100:.2f}]\n"
            f"* Expected binomial sigma: %{sigma*100:.2f}\n\n"
            f"**Uncertainty gate**\n\n"
            f"* Suspended: {st.session_state.suspended}/"
            f"{st.session_state.counter} "
            f"(%{100*st.session_state.suspended/st.session_state.counter:.1f})"
        )
        with g1:
            st.write("### Sliding-window accuracy")
            st.line_chart(pd.DataFrame({"accuracy": list(st.session_state.curve)}))
    else:
        info_box.info(f"Filling window... {n_samples}/30")

    with g2:
        st.write("### Confidence distribution")
        st.line_chart(pd.DataFrame(
            {"confidence": [conf for _, conf in st.session_state.window]}))

    time.sleep(REFRESH_RATE_SEC)
    st.rerun()