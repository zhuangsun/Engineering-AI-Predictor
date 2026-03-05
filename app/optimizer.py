"""
Multi-objective optimisation utilities.

Provides two optimisation strategies over the RF surrogate model:
  - Random-sampling Pareto filter  (fast, O(n) model evals)
  - NSGA-II evolutionary algorithm (thorough, iterative)

Both minimise weight and maximise strength (min –strength internally).
"""
import numpy as np
from app.services import model

# ── Shared helpers ────────────────────────────────────────────────────────────

def _pareto_mask(costs: np.ndarray) -> np.ndarray:
    """Boolean mask of non-dominated rows in `costs` (all objectives minimised)."""
    n = len(costs)
    is_efficient = np.ones(n, dtype=bool)
    for i in range(n):
        if not is_efficient[i]:
            continue
        dominated = (
            np.all(costs <= costs[i], axis=1) &
            np.any(costs < costs[i], axis=1)
        )
        dominated[i] = False
        if dominated.any():
            is_efficient[i] = False
    return is_efficient


def _dominates(a: np.ndarray, b: np.ndarray) -> bool:
    """True if `a` dominates `b` (all objectives minimised)."""
    return bool(np.all(a <= b) and np.any(a < b))


# ── NSGA-II internals ─────────────────────────────────────────────────────────

def _fast_non_dominated_sort(costs: np.ndarray) -> list[list[int]]:
    """Return list of Pareto fronts (front 0 = best) as index lists."""
    n = len(costs)
    dominated_set = [[] for _ in range(n)]   # S[i]: indices dominated by i
    domination_count = np.zeros(n, dtype=int)  # n_dom[i]: # points dominating i

    for i in range(n):
        for j in range(i + 1, n):
            if _dominates(costs[i], costs[j]):
                dominated_set[i].append(j)
                domination_count[j] += 1
            elif _dominates(costs[j], costs[i]):
                dominated_set[j].append(i)
                domination_count[i] += 1

    fronts: list[list[int]] = [[i for i in range(n) if domination_count[i] == 0]]
    k = 0
    while fronts[k]:
        next_front: list[int] = []
        for i in fronts[k]:
            for j in dominated_set[i]:
                domination_count[j] -= 1
                if domination_count[j] == 0:
                    next_front.append(j)
        k += 1
        fronts.append(next_front)

    return fronts[:-1]  # drop the empty sentinel


def _crowding_distance(costs: np.ndarray, front: list[int]) -> np.ndarray:
    """Crowding distance for a set of front indices into `costs`."""
    m = len(front)
    if m <= 2:
        return np.full(m, np.inf)
    dist = np.zeros(m)
    for obj in range(costs.shape[1]):
        vals = costs[front, obj]
        order = np.argsort(vals)
        dist[order[0]] = np.inf
        dist[order[-1]] = np.inf
        span = vals[order[-1]] - vals[order[0]]
        if span == 0:
            continue
        for k in range(1, m - 1):
            dist[order[k]] += (vals[order[k + 1]] - vals[order[k - 1]]) / span
    return dist


def _sbx_crossover(
    p1: np.ndarray, p2: np.ndarray, bounds: np.ndarray, eta: float = 15.0
) -> tuple[np.ndarray, np.ndarray]:
    """Simulated Binary Crossover (SBX)."""
    c1, c2 = p1.copy(), p2.copy()
    for i in range(len(p1)):
        if np.random.rand() > 0.5 or abs(p1[i] - p2[i]) < 1e-10:
            continue
        u = np.random.rand()
        beta = (
            (2 * u) ** (1 / (eta + 1))
            if u <= 0.5
            else (1 / (2 * (1 - u))) ** (1 / (eta + 1))
        )
        lo, hi = bounds[i]
        c1[i] = np.clip(0.5 * ((p1[i] + p2[i]) - beta * abs(p2[i] - p1[i])), lo, hi)
        c2[i] = np.clip(0.5 * ((p1[i] + p2[i]) + beta * abs(p2[i] - p1[i])), lo, hi)
    return c1, c2


def _polynomial_mutation(
    x: np.ndarray, bounds: np.ndarray, eta: float = 20.0
) -> np.ndarray:
    """Polynomial mutation with per-gene probability 1/n_vars."""
    mutant = x.copy()
    prob = 1.0 / len(x)
    for i in range(len(x)):
        if np.random.rand() > prob:
            continue
        lo, hi = bounds[i]
        u = np.random.rand()
        delta_q = (
            (2 * u) ** (1 / (eta + 1)) - 1
            if u < 0.5
            else 1 - (2 * (1 - u)) ** (1 / (eta + 1))
        )
        mutant[i] = np.clip(mutant[i] + delta_q * (hi - lo), lo, hi)
    return mutant


def _tournament_select(
    pop: np.ndarray, rank: np.ndarray, crowd: np.ndarray
) -> np.ndarray:
    """Binary tournament: prefer lower rank, break ties with higher crowding."""
    i, j = np.random.choice(len(pop), 2, replace=False)
    if rank[i] < rank[j] or (rank[i] == rank[j] and crowd[i] > crowd[j]):
        return pop[i].copy()
    return pop[j].copy()


# ── Public optimisation functions ─────────────────────────────────────────────

def run_optimization() -> dict:
    """Simple single-objective extrema finder (kept for backward compat)."""
    samples = np.random.uniform(low=[1, 5, 2], high=[10, 20, 10], size=(500, 3))
    preds = model.predict(samples)
    weights, strengths = preds[:, 0], preds[:, 1]
    bw, bs = int(np.argmin(weights)), int(np.argmax(strengths))
    return {
        "min_weight_design": {
            "inputs": {"thickness": float(samples[bw, 0]), "length": float(samples[bw, 1]), "width": float(samples[bw, 2])},
            "weight": float(weights[bw]), "strength": float(strengths[bw]),
        },
        "max_strength_design": {
            "inputs": {"thickness": float(samples[bs, 0]), "length": float(samples[bs, 1]), "width": float(samples[bs, 2])},
            "weight": float(weights[bs]), "strength": float(strengths[bs]),
        },
    }


def run_pareto_optimization(bounds: dict, n_samples: int = 2000) -> dict:
    """Random-sampling Pareto filter over the surrogate model."""
    low  = [bounds["thickness_min"], bounds["length_min"], bounds["width_min"]]
    high = [bounds["thickness_max"], bounds["length_max"], bounds["width_max"]]
    samples = np.random.uniform(low=low, high=high, size=(n_samples, 3))
    preds = model.predict(samples)
    weights, strengths = preds[:, 0], preds[:, 1]
    costs = np.column_stack([weights, -strengths])
    mask = _pareto_mask(costs)
    order = np.argsort(weights[mask])
    ps, pw, pstr = samples[mask][order], weights[mask][order], strengths[mask][order]
    return {
        "pareto_front": [
            {"thickness": float(ps[i, 0]), "length": float(ps[i, 1]), "width": float(ps[i, 2]),
             "weight": float(pw[i]), "strength": float(pstr[i])}
            for i in range(len(order))
        ],
        "n_total_samples": n_samples,
        "n_pareto_points": int(mask.sum()),
    }


def run_ga_optimization(bounds: dict, pop_size: int = 100, n_generations: int = 50) -> dict:
    """
    NSGA-II multi-objective optimisation over the surrogate model.

    Operators
    ---------
    - SBX crossover        (η_c = 15)
    - Polynomial mutation  (η_m = 20, p_mut = 1/n_vars per gene)
    - Binary tournament selection on (Pareto rank, crowding distance)
    """
    var_bounds = np.array([
        [bounds["thickness_min"], bounds["thickness_max"]],
        [bounds["length_min"],    bounds["length_max"]],
        [bounds["width_min"],     bounds["width_max"]],
    ])

    def _eval(X: np.ndarray) -> np.ndarray:
        preds = model.predict(X)
        return np.column_stack([preds[:, 0], -preds[:, 1]])  # [weight, -strength]

    # Initialise
    pop = np.column_stack([
        np.random.uniform(var_bounds[k, 0], var_bounds[k, 1], pop_size)
        for k in range(3)
    ])
    costs = _eval(pop)

    for _ in range(n_generations):
        # Rank and crowding for current population
        fronts = _fast_non_dominated_sort(costs)
        rank   = np.empty(pop_size, dtype=int)
        crowd  = np.zeros(pop_size)
        for r, front in enumerate(fronts):
            idx = np.array(front)
            rank[idx] = r
            crowd[idx] = _crowding_distance(costs, front)

        # Generate offspring via tournament → SBX → mutation
        offspring: list[np.ndarray] = []
        while len(offspring) < pop_size:
            p1 = _tournament_select(pop, rank, crowd)
            p2 = _tournament_select(pop, rank, crowd)
            c1, c2 = _sbx_crossover(p1, p2, var_bounds)
            offspring.append(_polynomial_mutation(c1, var_bounds))
            if len(offspring) < pop_size:
                offspring.append(_polynomial_mutation(c2, var_bounds))

        # Merge parent + offspring, reduce to pop_size via NSGA-II selection
        merged_pop   = np.vstack([pop, np.array(offspring)])
        merged_costs = np.vstack([costs, _eval(np.array(offspring))])
        new_fronts   = _fast_non_dominated_sort(merged_costs)

        selected: list[int] = []
        for front in new_fronts:
            if len(selected) + len(front) <= pop_size:
                selected.extend(front)
            else:
                remaining = pop_size - len(selected)
                cd = _crowding_distance(merged_costs, front)
                top = np.array(front)[np.argsort(-cd)[:remaining]]
                selected.extend(top.tolist())
                break

        idx    = np.array(selected)
        pop    = merged_pop[idx]
        costs  = merged_costs[idx]

    # Extract final Pareto front
    mask  = _pareto_mask(costs)
    pp, pc = pop[mask], costs[mask]
    order = np.argsort(pc[:, 0])
    return {
        "pareto_front": [
            {"thickness": float(pp[i, 0]), "length": float(pp[i, 1]), "width": float(pp[i, 2]),
             "weight": float(pc[i, 0]),    "strength": float(-pc[i, 1])}
            for i in order
        ],
        "n_generations":  n_generations,
        "pop_size":        pop_size,
        "n_pareto_points": int(mask.sum()),
    }


def run_sensitivity(
    variable: str,
    fixed_thickness: float,
    fixed_length: float,
    fixed_width: float,
    sweep_min: float,
    sweep_max: float,
    n_points: int = 60,
) -> dict:
    """
    Sweep one design variable while holding the other two fixed.
    Returns arrays of weight and strength across the sweep range.
    """
    sweep = np.linspace(sweep_min, sweep_max, n_points)
    fixed = {"thickness": fixed_thickness, "length": fixed_length, "width": fixed_width}

    cols = {
        "thickness": np.column_stack([sweep,              np.full(n_points, fixed["length"]), np.full(n_points, fixed["width"])]),
        "length":    np.column_stack([np.full(n_points, fixed["thickness"]), sweep,              np.full(n_points, fixed["width"])]),
        "width":     np.column_stack([np.full(n_points, fixed["thickness"]), np.full(n_points, fixed["length"]), sweep]),
    }
    X    = cols[variable]
    pred = model.predict(X)
    return {
        "variable":     variable,
        "sweep_values": sweep.tolist(),
        "weight":       pred[:, 0].tolist(),
        "strength":     pred[:, 1].tolist(),
    }
