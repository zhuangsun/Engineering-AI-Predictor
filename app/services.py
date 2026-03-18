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
    deflection.  The std serves as a proxy for surrogate-model uncertainty.
    """
    X = np.array([[data.h, data.l, data.t, data.b]])

    mean_pred  = model.predict(X)[0]                                            # (2,)
    tree_preds = np.array([tree.predict(X)[0] for tree in model.estimators_])  # (n_trees, 2)
    std_pred   = tree_preds.std(axis=0)

    return {
        "cost":           float(mean_pred[0]),
        "cost_std":       float(std_pred[0]),
        "deflection":     float(mean_pred[1]),
        "deflection_std": float(std_pred[1]),
    }
