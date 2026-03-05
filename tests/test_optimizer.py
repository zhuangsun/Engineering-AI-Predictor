"""Tests for Pareto optimizers, NSGA-II, and sensitivity analysis."""
import numpy as np
import pytest
from fastapi.testclient import TestClient
from app.main import app
from app.optimizer import _pareto_mask, run_pareto_optimization, run_ga_optimization, run_sensitivity

client = TestClient(app)

FULL_BOUNDS = {
    "thickness_min": 1.0, "thickness_max": 10.0,
    "length_min":    5.0, "length_max":    20.0,
    "width_min":     2.0, "width_max":     10.0,
}


# ── _pareto_mask ──────────────────────────────────────────────────────────────

class TestParetoMask:
    def test_single_point_is_efficient(self):
        assert _pareto_mask(np.array([[1.0, 2.0]])).all()

    def test_dominated_point_removed(self):
        mask = _pareto_mask(np.array([[1.0, 1.0], [2.0, 2.0]]))
        assert mask[0] and not mask[1]

    def test_trade_off_both_efficient(self):
        assert _pareto_mask(np.array([[1.0, 3.0], [3.0, 1.0]])).all()

    def test_identical_points_kept(self):
        assert _pareto_mask(np.array([[2.0, 2.0], [2.0, 2.0]])).all()

    def test_output_shape_and_dtype(self):
        mask = _pareto_mask(np.random.rand(50, 2))
        assert mask.shape == (50,) and mask.dtype == bool


# ── Random-sampling optimizer ─────────────────────────────────────────────────

class TestRunParetoOptimization:
    def test_keys(self):
        r = run_pareto_optimization(FULL_BOUNDS, n_samples=200)
        assert {"pareto_front", "n_total_samples", "n_pareto_points"} <= r.keys()

    def test_n_samples(self):
        assert run_pareto_optimization(FULL_BOUNDS, n_samples=300)["n_total_samples"] == 300

    def test_non_empty(self):
        r = run_pareto_optimization(FULL_BOUNDS, n_samples=200)
        assert r["n_pareto_points"] > 0 and len(r["pareto_front"]) == r["n_pareto_points"]

    def test_sorted_by_weight(self):
        weights = [p["weight"] for p in run_pareto_optimization(FULL_BOUNDS, n_samples=400)["pareto_front"]]
        assert weights == sorted(weights)

    def test_designs_within_bounds(self):
        b = {"thickness_min":3,"thickness_max":7,"length_min":8,"length_max":15,"width_min":4,"width_max":8}
        for p in run_pareto_optimization(b, n_samples=300)["pareto_front"]:
            assert b["thickness_min"] <= p["thickness"] <= b["thickness_max"]
            assert b["length_min"]    <= p["length"]    <= b["length_max"]
            assert b["width_min"]     <= p["width"]     <= b["width_max"]


# ── NSGA-II optimizer ─────────────────────────────────────────────────────────

class TestRunGAOptimization:
    def test_keys(self):
        r = run_ga_optimization(FULL_BOUNDS, pop_size=20, n_generations=5)
        assert {"pareto_front", "n_generations", "pop_size", "n_pareto_points"} <= r.keys()

    def test_metadata_matches(self):
        r = run_ga_optimization(FULL_BOUNDS, pop_size=30, n_generations=8)
        assert r["pop_size"] == 30 and r["n_generations"] == 8

    def test_non_empty_front(self):
        r = run_ga_optimization(FULL_BOUNDS, pop_size=20, n_generations=5)
        assert r["n_pareto_points"] > 0

    def test_sorted_by_weight(self):
        weights = [p["weight"] for p in
                   run_ga_optimization(FULL_BOUNDS, pop_size=20, n_generations=5)["pareto_front"]]
        assert weights == sorted(weights)

    def test_each_design_has_required_fields(self):
        for p in run_ga_optimization(FULL_BOUNDS, pop_size=20, n_generations=5)["pareto_front"]:
            for k in ("thickness", "length", "width", "weight", "strength"):
                assert k in p


# ── Sensitivity analysis ──────────────────────────────────────────────────────

class TestRunSensitivity:
    def test_keys(self):
        r = run_sensitivity("thickness", 5, 12, 6, 1, 10)
        assert {"variable", "sweep_values", "weight", "strength"} <= r.keys()

    def test_n_points(self):
        r = run_sensitivity("length", 5, 12, 6, 5, 20, n_points=40)
        assert len(r["sweep_values"]) == 40 == len(r["weight"]) == len(r["strength"])

    def test_monotonic_weight_vs_length(self):
        """Longer beam → more weight (all else equal)."""
        r = run_sensitivity("length", 5, 12, 6, 5, 20)
        assert r["weight"][-1] > r["weight"][0]

    def test_monotonic_strength_vs_thickness(self):
        """Thicker beam → lower strength per the synthetic formula."""
        r = run_sensitivity("thickness", 5, 12, 6, 1, 10)
        assert r["strength"][0] > r["strength"][-1]


# ── API endpoints ─────────────────────────────────────────────────────────────

class TestEndpoints:
    def test_optimize_multi_200(self):
        assert client.post("/optimize_multi", json=FULL_BOUNDS).status_code == 200

    def test_optimize_ga_200(self):
        payload = {**FULL_BOUNDS, "pop_size": 20, "n_generations": 5}
        assert client.post("/optimize_ga", json=payload).status_code == 200

    def test_optimize_ga_returns_front(self):
        payload = {**FULL_BOUNDS, "pop_size": 20, "n_generations": 5}
        data = client.post("/optimize_ga", json=payload).json()
        assert isinstance(data["pareto_front"], list) and len(data["pareto_front"]) > 0

    def test_sensitivity_200(self):
        payload = {"variable": "width", "fixed_thickness": 5, "fixed_length": 12,
                   "fixed_width": 6, "sweep_min": 2, "sweep_max": 10}
        assert client.post("/sensitivity", json=payload).status_code == 200

    def test_sensitivity_invalid_sweep_range_422(self):
        payload = {"variable": "width", "fixed_thickness": 5, "fixed_length": 12,
                   "fixed_width": 6, "sweep_min": 10, "sweep_max": 2}
        assert client.post("/sensitivity", json=payload).status_code == 422

    def test_optimize_multi_missing_bound_422(self):
        incomplete = {k: v for k, v in FULL_BOUNDS.items() if k != "width_max"}
        assert client.post("/optimize_multi", json=incomplete).status_code == 422
