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
import numpy as np
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

X = np.column_stack([h, l, t, b])
y = np.column_stack([cost, deflection])

# ── Train / test split ────────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ── Model training ────────────────────────────────────────────────────────────
model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)

# ── Evaluation ────────────────────────────────────────────────────────────────
y_pred = model.predict(X_test)
r2  = r2_score(y_test, y_pred, multioutput="raw_values")
mae = mean_absolute_error(y_test, y_pred, multioutput="raw_values")

print(f"Train samples    : {len(X_train)}   Test samples : {len(X_test)}")
print(f"R²   — cost: {r2[0]:.4f}   deflection: {r2[1]:.4f}")
print(f"MAE  — cost: {mae[0]:.4f}   deflection: {mae[1]:.6f}")

# ── Serialise ─────────────────────────────────────────────────────────────────
os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/model.pkl")
print("Model saved to models/model.pkl")
