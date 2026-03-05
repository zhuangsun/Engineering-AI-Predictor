import numpy as np
from app.services import model


def _pareto_mask(costs: np.ndarray) -> np.ndarray:
    """
    Return a boolean mask of non-dominated points.

    A point i is non-dominated if no other point j satisfies:
        costs[j] <= costs[i] in all objectives  AND
        costs[j] <  costs[i] in at least one objective.

    `costs` columns must all be oriented as "minimize".
    """
    n = len(costs)
    is_efficient = np.ones(n, dtype=bool)
    for i in range(n):
        if not is_efficient[i]:
            continue
        # j dominates i when j is no worse in every dim and strictly better in ≥1
        dominated = (
            np.all(costs <= costs[i], axis=1) &
            np.any(costs < costs[i], axis=1)
        )
        dominated[i] = False
        if dominated.any():
            is_efficient[i] = False
    return is_efficient


def run_optimization():
    """Simple single-objective optimum finder (kept for backward compat)."""
    samples = np.random.uniform(low=[1, 5, 2], high=[10, 20, 10], size=(500, 3))
    predictions = model.predict(samples)
    weights, strengths = predictions[:, 0], predictions[:, 1]

    bw, bs = int(np.argmin(weights)), int(np.argmax(strengths))
    return {
        "min_weight_design": {
            "inputs": {
                "thickness": float(samples[bw][0]),
                "length": float(samples[bw][1]),
                "width": float(samples[bw][2]),
            },
            "weight": float(weights[bw]),
            "strength": float(strengths[bw]),
        },
        "max_strength_design": {
            "inputs": {
                "thickness": float(samples[bs][0]),
                "length": float(samples[bs][1]),
                "width": float(samples[bs][2]),
            },
            "weight": float(weights[bs]),
            "strength": float(strengths[bs]),
        },
    }


def run_pareto_optimization(bounds: dict, n_samples: int = 2000) -> dict:
    """
    Sample the design space uniformly, evaluate the surrogate model, and
    return the Pareto-optimal front for the two-objective problem:
        - Minimize weight
        - Maximize strength  (internally: minimize –strength)

    Parameters
    ----------
    bounds : dict with keys thickness_min/max, length_min/max, width_min/max
    n_samples : number of random samples drawn from the feasible space

    Returns
    -------
    dict with 'pareto_front' list sorted by ascending weight,
    plus metadata (n_total_samples, n_pareto_points).
    """
    low = [bounds["thickness_min"], bounds["length_min"], bounds["width_min"]]
    high = [bounds["thickness_max"], bounds["length_max"], bounds["width_max"]]

    samples = np.random.uniform(low=low, high=high, size=(n_samples, 3))
    preds = model.predict(samples)
    weights, strengths = preds[:, 0], preds[:, 1]

    # Cost matrix: [weight, -strength] — both minimized
    costs = np.column_stack([weights, -strengths])
    mask = _pareto_mask(costs)

    pareto_samples = samples[mask]
    pareto_weights = weights[mask]
    pareto_strengths = strengths[mask]

    # Sort by ascending weight for a clean Pareto curve
    order = np.argsort(pareto_weights)

    return {
        "pareto_front": [
            {
                "thickness": float(pareto_samples[i, 0]),
                "length": float(pareto_samples[i, 1]),
                "width": float(pareto_samples[i, 2]),
                "weight": float(pareto_weights[i]),
                "strength": float(pareto_strengths[i]),
            }
            for i in order
        ],
        "n_total_samples": n_samples,
        "n_pareto_points": int(mask.sum()),
    }
