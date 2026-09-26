# Nexus-Vacuum

Simulation and machine-learning pipeline for predictive maintenance of in-space manufacturing actuators (an LTP02 thermal printer mechanism and a 25BY24 stepper motor) under simulated Low Earth Orbit (LEO) vacuum conditions.

This repository accompanies the paper *"An Autonomous Predictive Maintenance and Visual Telemetry Framework for In-Space Manufacturing Actuators Under Simulated LEO Vacuum Conditions"* (Nisanur Yıldırım).

## Highlights

- Physics-based thermomechanical + torque-stress simulation of a stepper-motor actuator under LEO vacuum conditions, with stochastic sensor noise injected on top.
- Random Forest fault classifier (Healthy / Thermal / Missing-Step / Voltage-Drop), benchmarked against a deterministic threshold rule, logistic regression, and a decision tree, with a formal McNemar significance test.
- Explicit Bayes-ceiling analysis: quantifies how much of the model's remaining error is due to injected label noise rather than model capacity.
- Uncertainty-aware decision-suspension policy: autonomous actions are withheld when model confidence drops below a tunable threshold, raising effective accuracy at the cost of deferring some decisions.
- Live Streamlit dashboard for real-time sliding-window telemetry monitoring.

> **Note:** All data in this repository is synthetic. It is generated from a physics-based thermomechanical simulation with injected stochastic noise; it has not been validated against real flight or thermal-vacuum (TVAC) test data. See the paper's Limitations section for details.

## Repository contents

| File | Description |
|---|---|
| `01_create_data.py` | Telemetry Generation Module. Simulates actuator thermal and torque behavior, injects Gaussian sensor noise, and generates sigmoid-based probabilistic fault labels. Produces `nexus_vakum_veri.csv`. |
| `02_baseline.py` | Machine Learning Training and Baseline Comparison Engine. Trains the Random Forest classifier, runs the deterministic-threshold / logistic regression / decision tree baselines, performs 10-fold cross-validation and the McNemar significance test, and computes the Bayes ceiling. Produces `nexus_model.pkl`, `model_columns.pkl`, and `02_results.json`. |
| `app.py` | Live Monitoring Interface (Streamlit). Runs the real-time sliding-window simulation, applies the confidence-based decision-suspension policy, and displays live accuracy / confidence charts. |
| `nexus_vakum_config.json` | Physical and simulation configuration parameters (thermal, torque, and labeling constants) shared by the generator and the app. |
| `nexus_vakum_veri.csv` | Example generated telemetry dataset. |
| `nexus_model.pkl` / `model_columns.pkl` | Trained Random Forest model and the feature-column order it expects. |
| `02_results.json` | Cached evaluation metrics (used by the paper's result tables). |

## Setup

```bash
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

`requirements.txt` (create this file with):
```
numpy
pandas
scikit-learn
streamlit
joblib
```

*(Pin exact versions here once you confirm which versions you used — this matters for reproducibility.)*

## Running the pipeline

Run the scripts in this order:

```bash
python 01_create_data.py     # generates nexus_vakum_veri.csv
python 02_baseline.py        # trains the model, runs baselines, saves nexus_model.pkl
streamlit run app.py         # launches the live monitoring dashboard
```

## Reproducing the paper's numbers

- Table I / II (classification performance, confusion matrix) and the 10-fold CV results are produced by `02_baseline.py`.
- Table III (baseline comparison, McNemar test, Bayes ceiling) is also produced by `02_baseline.py` — see `02_results.json` for the cached values.
- Table IV (confidence-gate effect) and the sliding-window accuracy plot are produced live by `app.py`.

## Known limitations (see paper for full discussion)

- Entirely synthetic data; no real hardware or TVAC validation.
- The Bayes ceiling comparison in the paper is a self-consistency check on the simulation design (both the label noise and labeling function are fixed by the authors), not independent evidence of real-world generalization.
- Overall accuracy is statistically indistinguishable from a simple deterministic threshold rule (McNemar p = 0.82); the model's value is in rare-class macro-F1 and in enabling the confidence-based decision-suspension policy.

## Citation

If you use this code, please cite the accompanying paper (see repository description / linked arXiv entry once available).

## License

*(No license file is currently included — add one, e.g. MIT or Apache-2.0, if you want others to be able to reuse this code.)*
