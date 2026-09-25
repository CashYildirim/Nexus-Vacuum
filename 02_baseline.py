import json
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, f1_score, classification_report,
                             confusion_matrix)
from scipy.stats import chi2

CLASSES = ["Healthy", "Thermal", "Step Loss", "Voltage"]
DERIVED_FEATURES = ["Calculated_Temperature_C", "Torque_Stress_Coefficient"]

# Deterministic thresholds
THRESHOLDS = {"T_C": 100.0, "S": 0.95, "V": 5.8, "I": 0.35}


def threshold_rule(X, thresholds=THRESHOLDS):
    """Classic deterministic logic used in space systems.
    Priority order: thermal > mechanical > electrical."""
    T = X["Calculated_Temperature_C"].values
    S = X["Torque_Stress_Coefficient"].values
    V = X["Operating_Voltage_V"].values
    I = X["Phase_Current_A"].values
    y = np.zeros(len(X), dtype=int)
    y[(V < thresholds["V"]) & (I > thresholds["I"])] = 3
    y[S > thresholds["S"]] = 2
    y[T > thresholds["T_C"]] = 1
    return y


def bootstrap_ci(y_true, y_pred, metric=accuracy_score, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    N = len(y_true)
    scores = [metric(y_true[i], y_pred[i]) for i in
              (rng.integers(0, N, N) for _ in range(n))]
    return np.percentile(scores, [2.5, 97.5])


def mcnemar(y_true, pred_a, pred_b):
    """Compares error patterns of models A and B.
    b = A correct & B wrong, c = A wrong & B correct."""
    a_ok = pred_a == y_true
    b_ok = pred_b == y_true
    b = int((a_ok & ~b_ok).sum())
    c = int((~a_ok & b_ok).sum())
    if b + c == 0:
        return b, c, 1.0
    stat = (abs(b - c) - 1) ** 2 / (b + c)
    return b, c, float(1 - chi2.cdf(stat, 1))


def false_alarm_rate(y_true, y_pred):
    """Rate of declaring fault when actually healthy.
    This is the direct cost of unnecessary shutdowns in space missions."""
    healthy = y_true == 0
    return float((y_pred[healthy] != 0).mean())


def missed_fault_rate(y_true, y_pred):
    """Rate of predicting 'healthy' when a fault exists. Safety-critical."""
    faulty = y_true != 0
    return float((y_pred[faulty] == 0).mean())


def main():
    df = pd.read_csv("nexus_vakum_veri.csv")
    X = df.drop(columns=["Fault_Status"])
    y = df["Fault_Status"].values

    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    models = {
        "Logistic Regression": make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=2000, class_weight="balanced",
                               random_state=42)),
        "Decision Tree (d=3)": DecisionTreeClassifier(
            max_depth=3, class_weight="balanced", random_state=42),
        "Random Forest": RandomForestClassifier(
            n_estimators=100, max_depth=12, class_weight="balanced",
            random_state=42, n_jobs=-1),
    }

    predictions = {"Threshold Rule (deterministic)": threshold_rule(Xte)}
    for name, m in models.items():
        m.fit(Xtr, ytr)
        predictions[name] = m.predict(Xte)

    # =================================================================
    # TABLE A: BASELINE COMPARISON
    # =================================================================
    print("=" * 84)
    print("TABLE A  -  BASELINE COMPARISON (test n=%d)" % len(yte))
    print("=" * 84)
    print(f"{'Method':<30}{'Accuracy':>12}{'95% CI':>18}"
          f"{'Macro F1':>11}{'False Alarm':>14}{'Missed Fault':>14}")
    print("-" * 84)

    results = {}
    for name, p in predictions.items():
        acc = accuracy_score(yte, p)
        lo, hi = bootstrap_ci(yte, p)
        mf1 = f1_score(yte, p, average="macro", zero_division=0)
        fa = false_alarm_rate(yte, p)
        ka = missed_fault_rate(yte, p)
        results[name] = {"accuracy": round(acc, 4),
                         "ci95": [round(lo, 4), round(hi, 4)],
                         "macro_f1": round(mf1, 4),
                         "false_alarm": round(fa, 4),
                         "missed_fault": round(ka, 4)}
        print(f"{name:<30}{acc*100:>11.2f}%"
              f"{f'[{lo*100:.1f}, {hi*100:.1f}]':>18}"
              f"{mf1:>11.3f}{fa*100:>13.2f}%{ka*100:>13.2f}%")
    print("=" * 84)

    # =================================================================
    # TABLE B: McNEMAR TEST
    # =================================================================
    print("\nTABLE B  -  McNEMAR TEST (vs Random Forest)")
    print("-" * 70)
    rf = predictions["Random Forest"]
    mcn = {}
    for name, p in predictions.items():
        if name == "Random Forest":
            continue
        b, c, pv = mcnemar(yte, p, rf)
        mcn[name] = {"only_A_correct": b, "only_RF_correct": c,
                     "p_value": round(pv, 6)}
        signif = "SIGNIFICANT" if pv < 0.05 else "not significant"
        print(f"  {name:<30} b={b:4d}  c={c:4d}  p={pv:.2e}  -> {signif}")

    # =================================================================
    # TABLE C: CONFUSION MATRIX
    # =================================================================
    print("\nTABLE C  -  CONFUSION MATRIX (Random Forest)")
    print("-" * 70)
    cm = confusion_matrix(yte, rf, labels=[0, 1, 2, 3])
    print(f"{'actual \\ predicted':<18}" + "".join(f"{s:>15}" for s in CLASSES))
    for i, s in enumerate(CLASSES):
        print(f"{s:<18}" + "".join(f"{v:>15d}" for v in cm[i]))

    print("\n" + classification_report(yte, rf, labels=[0, 1, 2, 3],
                                       target_names=CLASSES, zero_division=0,
                                       digits=3))

    # =================================================================
    # TABLE D: LEAKAGE ABLATION
    # =================================================================
    print("=" * 84)
    print("TABLE D  -  LEAKAGE ABLATION")
    print("  Derived features are direct inputs to the labeling logic.")
    print("  How much does performance drop when they are removed?")
    print("=" * 84)
    ablation = {}
    for name, cols in [("All features", list(X.columns)),
                       ("Derived REMOVED",
                        [c for c in X.columns if c not in DERIVED_FEATURES])]:
        m = RandomForestClassifier(n_estimators=100, max_depth=12,
                                   class_weight="balanced",
                                   random_state=42, n_jobs=-1)
        m.fit(Xtr[cols], ytr)
        acc = m.score(Xte[cols], yte)
        mf1 = f1_score(yte, m.predict(Xte[cols]), average="macro",
                       zero_division=0)
        ablation[name] = {"accuracy": round(acc, 4), "macro_f1": round(mf1, 4),
                          "feature_count": len(cols)}
        print(f"  {name:<26} n_feat={len(cols)}  accuracy=%{acc*100:.2f}  "
              f"macroF1={mf1:.3f}")
    diff = (ablation["All features"]["accuracy"]
            - ablation["Derived REMOVED"]["accuracy"])
    print(f"\n  -> Contribution of derived features: {diff*100:.2f} points")
    print("     Discuss this difference in the 'Limitations' section of your paper.")

    # =================================================================
    # TABLE E: UNCERTAINTY GATE
    # =================================================================
    print("\n" + "=" * 84)
    print("TABLE E  -  UNCERTAINTY GATE")
    print("  Policy: if max(predict_proba) < threshold, autonomous decision is suspended")
    print("=" * 84)
    rf_model = models["Random Forest"]
    prob = rf_model.predict_proba(Xte).max(axis=1)
    print(f"{'Threshold':>10}{'Suspended':>16}{'Remaining Accuracy':>20}"
          f"{'Suspended Accuracy':>22}")
    print("-" * 70)
    gate = {}
    for t in [0.0, 0.60, 0.70, 0.80, 0.90, 0.95]:
        passed = prob >= t
        if passed.sum() == 0:
            continue
        acc_g = accuracy_score(yte[passed], rf[passed])
        acc_a = (accuracy_score(yte[~passed], rf[~passed])
                 if (~passed).sum() > 0 else float("nan"))
        gate[str(t)] = {"suspended_ratio": round(1 - passed.mean(), 4),
                        "remaining_accuracy": round(acc_g, 4),
                        "suspended_accuracy": None if np.isnan(acc_a)
                        else round(acc_a, 4)}
        print(f"{t:>10.2f}{(1-passed.mean())*100:>15.2f}%{acc_g*100:>19.2f}%"
              f"{('  -' if np.isnan(acc_a) else f'{acc_a*100:.2f}%'):>22}")
    print("\n  Filtering uncertainty improves accuracy at the cost of suspended samples.")

    # =================================================================
    # CROSS VALIDATION
    # =================================================================
    cv = cross_val_score(
        RandomForestClassifier(n_estimators=100, max_depth=12,
                               class_weight="balanced", random_state=42,
                               n_jobs=-1),
        Xtr, ytr, cv=StratifiedKFold(10, shuffle=True, random_state=42))
    print("\n" + "=" * 84)
    print(f"10-fold cross validation: %{cv.mean()*100:.2f} (+/- %{cv.std()*100:.2f})")
    rf_model.fit(Xtr, ytr)
    print(f"Training accuracy       : %{rf_model.score(Xtr, ytr)*100:.2f}")
    print(f"Test accuracy           : %{rf_model.score(Xte, yte)*100:.2f}")
    print(f"Train-Test difference   : "
          f"{(rf_model.score(Xtr, ytr)-rf_model.score(Xte, yte))*100:.2f} points")
    print("=" * 84)

    with open("02_results.json", "w") as f:
        json.dump({"baseline": results, "mcnemar": mcn,
                   "confusion_matrix": cm.tolist(),
                   "ablation": ablation, "uncertainty_gate": gate,
                   "cv_mean": round(float(cv.mean()), 4),
                   "cv_std": round(float(cv.std()), 4)},
                  f, indent=2, ensure_ascii=False)
    print("\nResults saved -> 02_results.json")

    joblib.dump(rf_model, "nexus_model.pkl")
    joblib.dump(list(X.columns), "model_columns.pkl")
    print("Model saved             -> nexus_model.pkl")
    print("Column order saved      -> model_columns.pkl")
    print(f"  Feature order: {list(X.columns)}")


if __name__ == "__main__":
    main()