import json
import numpy as np
import pandas as pd

SIGMA_SB = 5.670374419e-8

CFG = {
    "seed": 42,
    "n_sample": 10000,

    # --- Thermal Actuator ---
    "mass_kg": 0.150,
    "specific_heat_J_kgK": 460.0,
    "surface_area_m2": 0.0042,
    "emissivity": 0.80,
    "view_factor": 0.15,
    "space_temperature_K": 200.0,
    "time_step_s": 15.0,
    "max_task_duration_s": 10800.0,

    # --- Vacuum / Convection ---
    "pressure_Torr": [1e-6, 1e-2],
    "h_reference_W_m2K": 8.5,
    "pressure_half_Torr": 1.0,

    # --- Mechanical ---
    "holding_torque_mNm": 90.0,
    "corner_frequency_PPS": 1800.0,
    "nominal_voltage_V": 9.0,
    "torque_derating_1_K": 0.0035,
    "friction_base_mNm": 4.0,
    "friction_gain_mNm": 14.0,
    "static_load_mNm": 8.0,

    # --- Labeling ---
    "T_sigmoid_center_C": 100.0, "T_sigmoid_slope": 4.0, "T_door_C": 80.0,
    "S_sigmoid_center": 0.95, "S_sigmoid_slope": 0.07, "S_door": 0.75,
    "V_critical_V": 5.8, "I_critical_A": 0.35, "p_voltage": 0.80,

    "label_noise": 0.05,

    # --- Measurement Noise ---
    "thermistor_sigma_C": 1.5,
    "torque_sigma": 0.02,

    # --- Task Profile Weights ---
    "profile_weights": [0.50, 0.20, 0.18, 0.12],
    "profile_names": ["Nominal", "Thermal_Load", "Mechanical_Load", "Power_Anomaly"],
}


def generate_data(cfg=CFG, save=True):
    rng = np.random.default_rng(cfg["seed"])
    N = cfg["n_sample"]

    # =================================================================
    # 1. TASK PROFILE MIXTURE
    # =================================================================
    prof = rng.choice(4, N, p=cfg["profile_weights"])
    U = lambda a, b: rng.uniform(a, b, N)
    S = lambda a, b, c, d: np.select(
        [prof == 0, prof == 1, prof == 2, prof == 3], [a, b, c, d])

    ambient_C = S(U(-50, 30), U(20, 80), U(-20, 50), U(-40, 40))
    voltage = S(U(6.8, 9.5), U(6.5, 9.5), U(6.0, 9.0), U(5.0, 6.3))
    current = S(U(.20, .34), U(.34, .45), U(.28, .42), U(.34, .45))
    resistance = S(U(9, 16), U(15, 20), U(11, 19), U(9, 20))
    pps = S(U(100, 1400), U(300, 1800), U(2000, 3200), U(200, 2000))

    pressure = 10 ** rng.uniform(*np.log10(cfg["pressure_Torr"]), N)

    # ---- LATENT variables: NOT written to CSV ----
    print_density = S(U(.10, .45), U(.70, 1.0), U(.45, .95), U(.20, .70))
    lubricant = S(rng.beta(2, 6, N), rng.beta(3, 4, N),
                  rng.beta(6, 2, N), rng.beta(2, 5, N))
    R_conduction = S(U(80, 220), U(200, 450), U(100, 300), U(80, 300))
    task_duration = S(U(600, 3600), U(2400, 10800), U(1200, 7200), U(600, 5400))

    # =================================================================
    # 2. TRANSIENT THERMAL INTEGRATION
    # =================================================================
    mc = cfg["mass_kg"] * cfg["specific_heat_J_kgK"]
    A = cfg["surface_area_m2"]
    epsF = cfg["emissivity"] * cfg["view_factor"]
    T_space4 = cfg["space_temperature_K"] ** 4
    dt = cfg["time_step_s"]

    h = cfg["h_reference_W_m2K"] * pressure / (pressure + cfg["pressure_half_Torr"])
    Q_joule = (current ** 2) * resistance * print_density          # W

    T0 = ambient_C + 273.15
    T = T0.copy()
    for k in range(int(cfg["max_task_duration_s"] / dt)):
        active = (k * dt) < task_duration
        dT = dt * (Q_joule
                   - epsF * SIGMA_SB * A * (T ** 4 - T_space4)
                   - h * A * (T - T0)
                   - (T - T0) / R_conduction) / mc
        T = T + np.where(active, dT, 0.0)

    temperature_C = T - 273.15
    temperature_measured = temperature_C + rng.normal(0, cfg["thermistor_sigma_C"], N)

    # =================================================================
    # 3. TORQUE BALANCE
    # =================================================================
    derating = np.clip(
        1.0 - cfg["torque_derating_1_K"] * np.maximum(temperature_C - 25.0, 0),
        0.25, 1.0)
    v_factor = np.clip(voltage / cfg["nominal_voltage_V"], 0.4, 1.0)

    torque_pullout = (cfg["holding_torque_mNm"] * derating * v_factor
                      / np.sqrt(1.0 + (pps / cfg["corner_frequency_PPS"]) ** 2))

    friction = (cfg["friction_base_mNm"] + cfg["friction_gain_mNm"]
                * lubricant * (1.0 + 0.004 * np.maximum(temperature_C - 25, 0)))
    torque_load = cfg["static_load_mNm"] + friction + 5.0 * print_density

    torque_stress = torque_load / np.maximum(torque_pullout, 1e-3)
    torque_measured = torque_stress + rng.normal(0, cfg["torque_sigma"], N)

    # =================================================================
    # 4. PROBABILISTIC LABELING
    # =================================================================
    p_t = np.where(temperature_C > cfg["T_door_C"],
                   1 / (1 + np.exp(-(temperature_C - cfg["T_sigmoid_center_C"])
                                   / cfg["T_sigmoid_slope"])), 0.0)
    p_s = np.where(torque_stress > cfg["S_door"],
                   1 / (1 + np.exp(-(torque_stress - cfg["S_sigmoid_center"])
                                   / cfg["S_sigmoid_slope"])), 0.0)
    p_v = np.where((voltage < cfg["V_critical_V"]) & (current > cfg["I_critical_A"]),
                   cfg["p_voltage"], 0.0)

    P = np.stack([np.zeros(N), p_t, p_s, p_v], axis=1)
    P[:, 0] = np.clip(1.0 - P[:, 1:].max(axis=1), 0.0, None)
    P = P / P.sum(axis=1, keepdims=True)

    label = np.array([rng.choice(4, p=P[i]) for i in range(N)])

    # ---- SINGLE label noise block ----
    clean = label.copy()
    n_flip = int(N * cfg["label_noise"])
    idx = rng.choice(N, size=n_flip, replace=False)
    label[idx] = rng.integers(0, 4, size=n_flip)
    actual_flips = int((label != clean).sum())

    # =================================================================
    # 5. SAVING
    # =================================================================
    df = pd.DataFrame({
        "Ambient_Temperature_C": ambient_C,
        "Vacuum_Pressure_Torr": pressure,
        "Operating_Voltage_V": voltage,
        "Phase_Current_A": current,
        "Phase_Resistance_Ohm": resistance,
        "Motor_Speed_PPS": pps,
        "Calculated_Temperature_C": temperature_measured,
        "Torque_Stress_Coefficient": torque_measured,
        "Fault_Status": label,
    })

    bayes_raw = float(P.max(axis=1).mean())
    bayes_noisy = ((1 - cfg["label_noise"]) * bayes_raw
                   + cfg["label_noise"] * 0.25)

    report = {
        "config": cfg,
        "class_distribution": np.bincount(label, minlength=4).tolist(),
        "profile_distribution": np.bincount(prof, minlength=4).tolist(),
        "label_noise_target": n_flip,
        "label_noise_actual": actual_flips,
        "bayes_ceiling_label_noise_free": round(bayes_raw, 4),
        "bayes_ceiling_label_noisy": round(bayes_noisy, 4),
        "temperature_C": {"min": round(float(temperature_C.min()), 1),
                           "max": round(float(temperature_C.max()), 1),
                           "mean": round(float(temperature_C.mean()), 1)},
        "torque_stress": {"min": round(float(torque_stress.min()), 3),
                           "max": round(float(torque_stress.max()), 3),
                           "mean": round(float(torque_stress.mean()), 3)},
        "latent_variables": ["print_density", "lubricant_degradation",
                            "R_conduction_K_W", "task_duration_s"],
    }

    if save:
        df.to_csv("nexus_vakum_veri.csv", index=False)
        with open("nexus_vakum_config.json", "w") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

    print("=" * 66)
    print("DATASET CREATED -> nexus_vakum_veri.csv")
    print("=" * 66)
    class_names = ["Healthy", "Thermal", "Step Loss", "Voltage"]
    for i, n in enumerate(report["class_distribution"]):
        print(f"  {i} {class_names[i]:14s} {n:5d}  (%{100*n/N:.1f})")
    print(f"\n  Temperature : {report['temperature_C']['min']} .. "
          f"{report['temperature_C']['max']} C  (mean {report['temperature_C']['mean']})")
    print(f"  Torque stress: {report['torque_stress']['min']} .. "
          f"{report['torque_stress']['max']}")
    print(f"  Label noise: target {n_flip}, actual {actual_flips}")
    print()
    print(f"  >> BAYES CEILING (noise-free labels) : %{bayes_raw*100:.2f}")
    print(f"  >> BAYES CEILING (noisy labels)      : %{bayes_noisy*100:.2f}")
    print("     No model can exceed this performance value.")
    print("     REPORT THIS NUMBER IN YOUR PAPER.")
    print("=" * 66)
    return df, report


if __name__ == "__main__":
    generate_data()