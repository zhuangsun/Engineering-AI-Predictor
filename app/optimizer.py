import numpy as np
import joblib
import os

MODEL_PATH = "models/model.pkl"

if not os.path.exists(MODEL_PATH):
    raise Exception("Model not found. Please run train.py first.")

model = joblib.load(MODEL_PATH)


def optimize(bounds, n_samples=5000):
    """
    bounds: list of tuples [(min,max), ...] for each variable
    """

    dim = len(bounds)
    samples = np.zeros((n_samples, dim))

    for i in range(dim):
        low, high = bounds[i]
        samples[:, i] = np.random.uniform(low, high, n_samples)

    predictions = model.predict(samples)

    best_index = np.argmax(predictions)

    best_parameters = samples[best_index]
    best_value = predictions[best_index]

    return best_parameters, best_value
