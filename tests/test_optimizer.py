"""
Tests for the Pareto optimizer — both the helper function and the API endpoint.

Requires the model to be trained first:  python train.py
"""
import numpy as np
import pytest
from fastapi.testclient import TestClient
from app.main import app
from app.optimizer import _pareto_mask, run_pareto_optimization

client = TestClient(app)

FULL_BOUNDS = {
    "thickness_min": 1.0, "thickness_max": 10.0,
    "length_min": 5.0,    "length_max": 20.0,
    "width_min": 2.0,     "width_max": 10.0,
}


# ── _pareto_mask unit tests ────────────────────────────────────────────────

class TestParetoMask:
    def test_single_point_is_efficient(self):
        costs = np.array([[1.0, 2.0]])
        assert _pareto_mask(costs).all()

    def test_dominated_point_removed(self):
        # Point [2, 2] is dominated by [1, 1]
        costs = np.array([[1.0, 1.0], [2.0, 2.0]])
        mask = _pareto_mask(costs)
        assert mask[0] is np.bool_(True)
        assert mask[1] is np.bool_(False)

    def test_trade_off_points_are_both_efficient(self):
        # [1, 3] vs [3, 1]: neither dominates the other
        costs = np.array([[1.0, 3.0], [3.0, 1.0]])
        assert _pareto_mask(costs).all()

    def test_identical_points_kept(self):
        # Two identical points — neither strictly dominates the other
        costs = np.array([[2.0, 2.0], [2.0, 2.0]])
        assert _pareto_mask(costs).all()

    def test_output_shape(self):
        costs = np.random.rand(50, 2)
        mask = _pareto_mask(costs)
        assert mask.shape == (50,)
        assert mask.dtype == bool


# ── run_pareto_optimization unit tests ────────────────────────────────────

class TestRunParetoOptimization:
    def test_returns_expected_keys(self):
        result = run_pareto_optimization(FULL_BOUNDS, n_samples=200)
        assert "pareto_front" in result
        assert "n_total_samples" in result
        assert "n_pareto_points" in result

    def test_n_samples_matches(self):
        result = run_pareto_optimization(FULL_BOUNDS, n_samples=300)
        assert result["n_total_samples"] == 300

    def test_pareto_front_non_empty(self):
        result = run_pareto_optimization(FULL_BOUNDS, n_samples=200)
        assert result["n_pareto_points"] > 0
        assert len(result["pareto_front"]) == result["n_pareto_points"]

    def test_pareto_front_sorted_by_weight(self):
        result = run_pareto_optimization(FULL_BOUNDS, n_samples=500)
        weights = [p["weight"] for p in result["pareto_front"]]
        assert weights == sorted(weights)

    def test_each_design_has_required_fields(self):
        result = run_pareto_optimization(FULL_BOUNDS, n_samples=200)
        for point in result["pareto_front"]:
            for key in ("thickness", "length", "width", "weight", "strength"):
                assert key in point, f"Missing key: {key}"

    def test_designs_within_bounds(self):
        bounds = {
            "thickness_min": 3.0, "thickness_max": 7.0,
            "length_min": 8.0,    "length_max": 15.0,
            "width_min": 4.0,     "width_max": 8.0,
        }
        result = run_pareto_optimization(bounds, n_samples=400)
        for p in result["pareto_front"]:
            assert bounds["thickness_min"] <= p["thickness"] <= bounds["thickness_max"]
            assert bounds["length_min"]    <= p["length"]    <= bounds["length_max"]
            assert bounds["width_min"]     <= p["width"]     <= bounds["width_max"]


# ── /optimize_multi endpoint tests ────────────────────────────────────────

class TestOptimizeMultiEndpoint:
    def test_returns_200(self):
        response = client.post("/optimize_multi", json=FULL_BOUNDS)
        assert response.status_code == 200

    def test_response_structure(self):
        data = client.post("/optimize_multi", json=FULL_BOUNDS).json()
        assert "pareto_front" in data
        assert isinstance(data["pareto_front"], list)
        assert len(data["pareto_front"]) > 0

    def test_missing_bound_returns_422(self):
        incomplete = {k: v for k, v in FULL_BOUNDS.items() if k != "width_max"}
        response = client.post("/optimize_multi", json=incomplete)
        assert response.status_code == 422
