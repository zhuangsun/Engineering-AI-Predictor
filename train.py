"""
Surrogate model training script.

Generates a synthetic structural dataset, trains a Random Forest regressor
with a train/test split, prints evaluation metrics, and serialises the model.
"""
import os
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
import joblib

# ── Synthetic dataset ─────────────────────────────────────────────────────────
np.random.seed(42)
N = 1000

thickness = np.random.uniform(1,  10, N)
length    = np.random.uniform(5,  20, N)
width     = np.random.uniform(2,  10, N)

weight   = thickness * length * width * 0.1
strength = 1000 / (thickness + 0.5) + width * 5

X = np.column_stack([thickness, length, width])
y = np.column_stack([weight, strength])

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

print(f"Train samples : {len(X_train)}   Test samples: {len(X_test)}")
print(f"R2   - weight: {r2[0]:.4f}   strength: {r2[1]:.4f}")
print(f"MAE  - weight: {mae[0]:.4f}   strength: {mae[1]:.4f}")

# ── Serialise ─────────────────────────────────────────────────────────────────
os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/model.pkl")
print("Model saved to models/model.pkl")
