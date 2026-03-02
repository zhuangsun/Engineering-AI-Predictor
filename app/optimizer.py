import numpy as np
from app.services import model


def run_optimization():
    # Random sampling
    samples = np.random.uniform(
        low=[1, 5, 2],
        high=[10, 20, 10],
        size=(200, 3)
    )

    predictions = model.predict(samples)

    weights = predictions[:, 0]
    strengths = predictions[:, 1]

    best_weight_idx = np.argmin(weights)
    best_strength_idx = np.argmax(strengths)

    return {
        "min_weight_design": {
            "inputs": samples[best_weight_idx].tolist(),
            "weight": float(weights[best_weight_idx]),
            "strength": float(strengths[best_weight_idx])
        },
        "max_strength_design": {
            "inputs": samples[best_strength_idx].tolist(),
            "weight": float(weights[best_strength_idx]),
            "strength": float(strengths[best_strength_idx])
        }
    }
