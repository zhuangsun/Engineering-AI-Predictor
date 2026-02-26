import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import r2_score, mean_squared_error
import joblib
import os


# ==========================================
# 1. Synthetic Engineering Data Generation
# ==========================================

def generate_data(n_samples=1000):
    """
    Simulated engineering dataset
    Input: 5 engineering parameters
    Output: 1 engineering response
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


# ==========================================
# 2. Train Model with Hyperparameter Search
# ==========================================

def train_model():

    print("Generating synthetic engineering data...")
    X, y = generate_data()

    # -------- Train / Test Split -------- #

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print("Starting hyperparameter search...")

    param_dist = {
        "n_estimators": [100, 200, 300],
        "max_depth": [None, 8, 12, 16],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
    }

    base_model = RandomForestRegressor(random_state=42, n_jobs=-1)

    search = RandomizedSearchCV(
        base_model,
        param_distributions=param_dist,
        n_iter=10,
        cv=3,
        scoring="r2",
        random_state=42,
        n_jobs=-1,
        verbose=1
    )

    search.fit(X_train, y_train)

    best_model = search.best_estimator_

    print("\nBest Hyperparameters:")
    print(search.best_params_)

    # -------- Evaluation -------- #

    train_preds = best_model.predict(X_train)
    test_preds = best_model.predict(X_test)

    train_r2 = r2_score(y_train, train_preds)
    test_r2 = r2_score(y_test, test_preds)

    test_mse = mean_squared_error(y_test, test_preds)

    print("\nModel Performance:")
    print(f"Train R2: {train_r2:.4f}")
    print(f"Test R2: {test_r2:.4f}")
    print(f"Test MSE: {test_mse:.6f}")

    # -------- Save Model -------- #

    os.makedirs("models", exist_ok=True)

    model_path = "models/model.pkl"

    joblib.dump(best_model, model_path)

    print(f"\nBest model saved to {model_path}")


# ==========================================
# 3. Main
# ==========================================

if __name__ == "__main__":
    train_model()