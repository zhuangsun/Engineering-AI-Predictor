import os
import joblib
import numpy as np

BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "model.pkl")

if not os.path.exists(MODEL_PATH):
    raise RuntimeError("Model not found. Run train.py first.")

model = joblib.load(MODEL_PATH)


def predict_design(data) -> dict:
    """
    Return mean prediction plus per-tree standard deviation for cost and
    deflection.  Deflection is stored in log-space by the model; we invert
    here and propagate uncertainty via the delta method:
      σ_δ ≈ exp(μ_log) · σ_log
    """
    X = np.array([[data.h, data.l, data.t, data.b]])

    mean_pred  = model.predict(X)[0]                                            # (2,)
    tree_preds = np.array([tree.predict(X)[0] for tree in model.estimators_])  # (n_trees, 2)

    # Cost: normal space — use std directly
    cost_mean = float(mean_pred[0])
    cost_std  = float(tree_preds[:, 0].std())

    # Deflection: log-space — invert mean, propagate std via delta method
    log_defl_mean = mean_pred[1]
    log_defl_std  = tree_preds[:, 1].std()
    defl_mean = float(np.exp(log_defl_mean))
    defl_std  = float(defl_mean * log_defl_std)

    return {
        "cost":           cost_mean,
        "cost_std":       cost_std,
        "deflection":     defl_mean,
        "deflection_std": defl_std,
    }
