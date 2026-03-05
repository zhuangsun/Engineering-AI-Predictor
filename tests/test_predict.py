"""
Tests for the /predict endpoint.

Requires the model to be trained first:  python train.py
"""
import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_predict_returns_weight_and_strength():
    response = client.post("/predict", json={"thickness": 5.0, "length": 10.0, "width": 6.0})
    assert response.status_code == 200
    data = response.json()
    assert "weight" in data
    assert "strength" in data


def test_predict_values_are_positive():
    response = client.post("/predict", json={"thickness": 3.0, "length": 8.0, "width": 4.0})
    data = response.json()
    assert data["weight"] > 0
    assert data["strength"] > 0


def test_predict_thicker_beam_weighs_more():
    """Physically: heavier cross-section → greater weight."""
    thin = client.post("/predict", json={"thickness": 2.0, "length": 10.0, "width": 5.0}).json()
    thick = client.post("/predict", json={"thickness": 9.0, "length": 10.0, "width": 5.0}).json()
    assert thick["weight"] > thin["weight"]


def test_predict_thicker_beam_is_weaker():
    """Physically: larger thickness reduces strength in the synthetic formula."""
    thin = client.post("/predict", json={"thickness": 2.0, "length": 10.0, "width": 5.0}).json()
    thick = client.post("/predict", json={"thickness": 9.0, "length": 10.0, "width": 5.0}).json()
    assert thin["strength"] > thick["strength"]


def test_predict_missing_field_returns_422():
    response = client.post("/predict", json={"thickness": 5.0, "length": 10.0})
    assert response.status_code == 422
