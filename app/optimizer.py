"""
Multi-objective optimisation utilities.

Provides two optimisation strategies over the RF surrogate model:
  - Random-sampling Pareto filter  (fast, O(n) model evals)
  - NSGA-II evolutionary algorithm (thorough, iterative)

Both minimise cost ($) and deflection (in) for the Welded Beam benchmark.
"""
import numpy as np
from app.services import model

# ── Welded Beam physics (used for feasibility enforcement) ─────────────────────
_P = 6_000   # applied load  [lbf]
_L = 14.0    # beam length   [in]


def _shear_stress(h, l, t, b):
    tau1 = _P / (np.sqrt(2) * h * l)
    M    = _P * (_L + l / 2)
    R    = np.sqrt(l**2 / 4 + ((h + t) / 2)**2)
    J    = 2 * np.sqrt(2) * h * l * (l**2 / 12 + ((h + t) / 2)**2)
    tau2 = M * R / J
    return np.sqrt(tau1**2 + 2 * tau1 * tau2 * (l / (2 * R)) + tau2**2)


def _bending_stress(h, l, t, b):
    return 504_000 / (t**2 * b)


def _deflection_physics(h, l, t, b):
    return 2.1952 / (t**3 * b)


def check_feasibility(h: float, l: float, t: float, b: float) -> dict:
    """Return constraint values and pass/fail status for a single design."""
    tau   = float(_shear_stress(h, l, t, b))
    sigma = float(_bending_stress(h, l, t, b))
    delta = float(_deflection_physics(h, l, t, b))
    return {
        "feasible": tau <= 13_600 and sigma <= 30_000 and delta <= 0.25 and h <= b,
        "constraints": {
            "shear_stress":   {"value": tau,   "limit": 13_600, "ok": tau   <= 13_600},
            "bending_stress": {"value": sigma, "limit": 30_000, "ok": sigma <= 30_000},
            "deflection":     {"value": delta, "limit": 0.25,   "ok": delta <= 0.25},
            "h_le_b":         {"value": h,     "limit": b,      "ok": h     <= b},
        },
    }


def _feasible_batch(X: np.ndarray) -> np.ndarray:
    """Vectorised feasibility check for a batch of designs (n, 4)."""
    h, l, t, b = X[:, 0], X[:, 1], X[:, 2], X[:, 3]
    return (
        (_shear_stress(h, l, t, b)   <= 13_600) &
        (_bending_stress(h, l, t, b) <= 30_000) &
        (_deflection_physics(h, l, t, b) <= 0.25) &
        (h <= b)
    )

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
    samples = np.random.uniform(low=[0.1, 0.1, 0.1, 0.1], high=[2.0, 10.0, 10.0, 2.0], size=(500, 4))
    preds = model.predict(samples)
    costs, deflections = preds[:, 0], np.exp(preds[:, 1])
    bc, bd = int(np.argmin(costs)), int(np.argmin(deflections))
    return {
        "min_cost_design": {
            "inputs": {"h": float(samples[bc, 0]), "l": float(samples[bc, 1]),
                       "t": float(samples[bc, 2]), "b": float(samples[bc, 3])},
            "cost": float(costs[bc]), "deflection": float(deflections[bc]),
        },
        "min_deflection_design": {
            "inputs": {"h": float(samples[bd, 0]), "l": float(samples[bd, 1]),
                       "t": float(samples[bd, 2]), "b": float(samples[bd, 3])},
            "cost": float(costs[bd]), "deflection": float(deflections[bd]),
        },
    }


def run_pareto_optimization(bounds: dict, n_samples: int = 2000) -> dict:
    """Random-sampling Pareto filter over the surrogate model."""
    low  = [bounds["h_min"], bounds["l_min"], bounds["t_min"], bounds["b_min"]]
    high = [bounds["h_max"], bounds["l_max"], bounds["t_max"], bounds["b_max"]]
    samples = np.random.uniform(low=low, high=high, size=(n_samples, 4))
    preds = model.predict(samples)
    costs_arr  = preds[:, 0]
    deflections = np.exp(preds[:, 1])       # invert log-transform from training
    # Remove infeasible samples before Pareto filtering
    feas = _feasible_batch(samples)
    samples, costs_arr, deflections = samples[feas], costs_arr[feas], deflections[feas]
    obj = np.column_stack([costs_arr, deflections])  # both minimised
    mask = _pareto_mask(obj)
    order = np.argsort(costs_arr[mask])
    ps = samples[mask][order]
    pc, pd = costs_arr[mask][order], deflections[mask][order]
    return {
        "pareto_front": [
            {"h": float(ps[i, 0]), "l": float(ps[i, 1]),
             "t": float(ps[i, 2]), "b": float(ps[i, 3]),
             "cost": float(pc[i]), "deflection": float(pd[i])}
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
        [bounds["h_min"], bounds["h_max"]],
        [bounds["l_min"], bounds["l_max"]],
        [bounds["t_min"], bounds["t_max"]],
        [bounds["b_min"], bounds["b_max"]],
    ])

    def _eval(X: np.ndarray) -> np.ndarray:
        preds = model.predict(X)
        cost  = preds[:, 0]
        defl  = np.exp(preds[:, 1])          # invert log-transform from training
        obj   = np.column_stack([cost, defl])
        # Penalise infeasible designs so they stay off the Pareto front
        infeasible = ~_feasible_batch(X)
        obj[infeasible] = 1e9
        return obj

    # Initialise
    pop = np.column_stack([
        np.random.uniform(var_bounds[k, 0], var_bounds[k, 1], pop_size)
        for k in range(4)
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
            {"h": float(pp[i, 0]), "l": float(pp[i, 1]),
             "t": float(pp[i, 2]), "b": float(pp[i, 3]),
             "cost": float(pc[i, 0]), "deflection": float(pc[i, 1])}
            for i in order
        ],
        "n_generations":   n_generations,
        "pop_size":         pop_size,
        "n_pareto_points":  int(mask.sum()),
    }


def run_sensitivity(
    variable: str,
    fixed_h: float,
    fixed_l: float,
    fixed_t: float,
    fixed_b: float,
    sweep_min: float,
    sweep_max: float,
    n_points: int = 60,
) -> dict:
    """
    Sweep one design variable while holding the other three fixed.
    Returns arrays of cost and deflection across the sweep range.
    """
    sweep = np.linspace(sweep_min, sweep_max, n_points)
    f = {"h": fixed_h, "l": fixed_l, "t": fixed_t, "b": fixed_b}
    fh, fl, ft, fb = np.full(n_points, f["h"]), np.full(n_points, f["l"]), \
                     np.full(n_points, f["t"]), np.full(n_points, f["b"])

    cols = {
        "h": np.column_stack([sweep, fl,    ft,    fb]),
        "l": np.column_stack([fh,    sweep, ft,    fb]),
        "t": np.column_stack([fh,    fl,    sweep, fb]),
        "b": np.column_stack([fh,    fl,    ft,    sweep]),
    }
    pred = model.predict(cols[variable])
    return {
        "variable":     variable,
        "sweep_values": sweep.tolist(),
        "cost":         pred[:, 0].tolist(),
        "deflection":   pred[:, 1].tolist(),
    }
