"""
Surrogate model training script — Welded Beam benchmark.

The Welded Beam is a standard engineering optimisation benchmark from the
structural design literature (Ragsdell & Phillips, 1976).

Variable     Symbol  Bounds         Unit
─────────────────────────────────────────
weld size      h     [0.1,  2.0]   in
weld length    l     [0.1, 10.0]   in
bar thickness  t     [0.1, 10.0]   in
bar height     b     [0.1,  2.0]   in

Objectives (both minimised)
  cost       = 1.10471·h²·l + 0.04811·t·b·(14 + l)   [$]
  deflection = 2.1952 / (t³·b)                         [in]

Feasibility constraints applied before training
  shear stress   τ  ≤ 13 600 psi
  bending stress σ  ≤ 30 000 psi
  deflection     δ  ≤ 0.25 in
  h              ≤  b
"""
import os
import json
import numpy as np
from datetime import date
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
import joblib

# ── Problem constants ──────────────────────────────────────────────────────────
P = 6_000   # applied load  [lbf]
L = 14.0    # beam length   [in]


def _cost(h, l, t, b):
    return 1.10471 * h**2 * l + 0.04811 * t * b * (14 + l)


def _deflection(h, l, t, b):
    return 2.1952 / (t**3 * b)


def _shear_stress(h, l, t, b):
    tau1 = P / (np.sqrt(2) * h * l)
    M    = P * (L + l / 2)
    R    = np.sqrt(l**2 / 4 + ((h + t) / 2)**2)
    J    = 2 * np.sqrt(2) * h * l * (l**2 / 12 + ((h + t) / 2)**2)
    tau2 = M * R / J
    return np.sqrt(tau1**2 + 2 * tau1 * tau2 * (l / (2 * R)) + tau2**2)


def _bending_stress(h, l, t, b):
    return 504_000 / (t**2 * b)


def _feasible(h, l, t, b):
    return (
        (_shear_stress(h, l, t, b)  <= 13_600) &
        (_bending_stress(h, l, t, b) <= 30_000) &
        (_deflection(h, l, t, b)    <=  0.25)  &
        (h <= b)
    )


# ── Dataset generation ─────────────────────────────────────────────────────────
np.random.seed(42)
N_raw = 10_000  # oversample; feasibility filter removes ~30–40 %

h = np.random.uniform(0.1,  2.0, N_raw)
l = np.random.uniform(0.1, 10.0, N_raw)
t = np.random.uniform(0.1, 10.0, N_raw)
b = np.random.uniform(0.1,  2.0, N_raw)

mask = _feasible(h, l, t, b)
h, l, t, b = h[mask], l[mask], t[mask], b[mask]
print(f"Feasible samples : {mask.sum()} / {N_raw}")

cost       = _cost(h, l, t, b)
deflection = _deflection(h, l, t, b)

# ── Add 2 % multiplicative noise so RF uncertainty estimates are meaningful ───
rng = np.random.default_rng(seed=123)
cost_noisy       = cost       * (1.0 + 0.02 * rng.standard_normal(len(cost)))
deflection_noisy = deflection * (1.0 + 0.02 * rng.standard_normal(len(deflection)))
deflection_noisy = np.clip(deflection_noisy, 1e-9, None)  # guard log domain

# Log-transform deflection: 2.1952/(t³·b) is linear in log-space → better R²
X = np.column_stack([h, l, t, b])
y = np.column_stack([cost_noisy, np.log(deflection_noisy)])

# ── Train / test split ────────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ── Model training ────────────────────────────────────────────────────────────
model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)

# ── Evaluation (convert back to original scale for reporting) ─────────────────
y_pred_log = model.predict(X_test)
y_pred_orig = y_pred_log.copy()
y_pred_orig[:, 1] = np.exp(y_pred_log[:, 1])

y_test_orig = y_test.copy()
y_test_orig[:, 1] = np.exp(y_test[:, 1])

r2  = r2_score(y_test_orig, y_pred_orig, multioutput="raw_values")
mae = mean_absolute_error(y_test_orig, y_pred_orig, multioutput="raw_values")

print(f"Train samples    : {len(X_train)}   Test samples : {len(X_test)}")
print(f"R²   — cost: {r2[0]:.4f}   deflection: {r2[1]:.4f}")
print(f"MAE  — cost: {mae[0]:.4f}   deflection: {mae[1]:.6f}")

# ── Serialise model ───────────────────────────────────────────────────────────
os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/model.pkl")
print("Model saved to models/model.pkl")

# ── Save training metadata ────────────────────────────────────────────────────
metadata = {
    "trained_at":               str(date.today()),
    "n_raw":                    N_raw,
    "n_feasible":               int(mask.sum()),
    "n_train":                  len(X_train),
    "n_test":                   len(X_test),
    "log_transform_deflection": True,
    "noise_pct":                0.02,
    "r2_cost":                  float(r2[0]),
    "r2_deflection":            float(r2[1]),
    "mae_cost":                 float(mae[0]),
    "mae_deflection":           float(mae[1]),
    "features":                 ["h", "l", "t", "b"],
    "targets":                  ["cost", "log_deflection"],
}
with open("models/model_info.json", "w") as f:
    json.dump(metadata, f, indent=2)
print("Metadata saved to models/model_info.json")
