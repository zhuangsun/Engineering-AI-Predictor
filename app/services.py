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
    Return mean prediction plus per-tree standard deviation for weight and
    strength.  The std serves as a proxy for surrogate-model uncertainty.
    """
    X = np.array([[data.thickness, data.length, data.width]])

    mean_pred  = model.predict(X)[0]                                       # (2,)
    tree_preds = np.array([t.predict(X)[0] for t in model.estimators_])   # (n_trees, 2)
    std_pred   = tree_preds.std(axis=0)

    return {
        "weight":       float(mean_pred[0]),
        "weight_std":   float(std_pred[0]),
        "strength":     float(mean_pred[1]),
        "strength_std": float(std_pred[1]),
    }
