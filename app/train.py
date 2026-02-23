import numpy as np
from sklearn.ensemble import RandomForestRegressor
import joblib
import os


def generate_data(n_samples=500):
    """
    Simulated engineering dataset
    5 input features -> 1 output
    """

    X = np.random.rand(n_samples, 5)

    y = (
        2 * X[:, 0]
        + 1.5 * X[:, 1] ** 2
        - 0.5 * X[:, 2]
        + 0.3 * X[:, 3]
        + np.sin(X[:, 4] * 3)
        + np.random.normal(0, 0.05, n_samples)
    )

    return X, y


def train_model():

    print("Generating data...")
    X, y = generate_data()

    print("Training model...")

    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=10,
        random_state=42
    )

    model.fit(X, y)

    os.makedirs("models", exist_ok=True)

    joblib.dump(model, "models/model.pkl")

    print("Model saved to models/model.pkl")


if __name__ == "__main__":
    train_model()