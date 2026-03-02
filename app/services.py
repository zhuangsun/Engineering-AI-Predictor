import joblib
import numpy as np
import os

# Absolute path safety
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "model.pkl")

if not os.path.exists(MODEL_PATH):
    raise Exception("Model not found. Run train.py first.")

model = joblib.load(MODEL_PATH)


def predict_design(data):
    X = np.array([[data.thickness, data.length, data.width]])
    prediction = model.predict(X)

    return {
        "weight": float(prediction[0][0]),
        "strength": float(prediction[0][1])
    }
