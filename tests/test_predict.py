"""Tests for /predict and /feature_importance endpoints."""
import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_predict_returns_all_fields():
    res = client.post("/predict", json={"thickness": 5.0, "length": 10.0, "width": 6.0})
    assert res.status_code == 200
    data = res.json()
    for key in ("weight", "weight_std", "strength", "strength_std"):
        assert key in data, f"Missing key: {key}"


def test_predict_values_are_positive():
    data = client.post("/predict", json={"thickness": 3.0, "length": 8.0, "width": 4.0}).json()
    assert data["weight"]   > 0
    assert data["strength"] > 0
    assert data["weight_std"]   >= 0
    assert data["strength_std"] >= 0


def test_predict_thicker_beam_weighs_more():
    thin  = client.post("/predict", json={"thickness": 2.0, "length": 10.0, "width": 5.0}).json()
    thick = client.post("/predict", json={"thickness": 9.0, "length": 10.0, "width": 5.0}).json()
    assert thick["weight"] > thin["weight"]


def test_predict_thicker_beam_is_weaker():
    thin  = client.post("/predict", json={"thickness": 2.0, "length": 10.0, "width": 5.0}).json()
    thick = client.post("/predict", json={"thickness": 9.0, "length": 10.0, "width": 5.0}).json()
    assert thin["strength"] > thick["strength"]


def test_predict_missing_field_returns_422():
    res = client.post("/predict", json={"thickness": 5.0, "length": 10.0})
    assert res.status_code == 422


# ── Feature importance ────────────────────────────────────────────────────────

def test_feature_importance_returns_200():
    res = client.get("/feature_importance")
    assert res.status_code == 200


def test_feature_importance_structure():
    data = client.get("/feature_importance").json()
    assert data["features"] == ["thickness", "length", "width"]
    assert len(data["importances"]) == 3


def test_feature_importance_sums_to_one():
    importances = client.get("/feature_importance").json()["importances"]
    assert abs(sum(importances) - 1.0) < 1e-6
