import numpy as np
from sklearn.ensemble import RandomForestRegressor
import joblib
import os

# Generate synthetic engineering dataset
np.random.seed(42)
n_samples = 500

thickness = np.random.uniform(1, 10, n_samples)
length = np.random.uniform(5, 20, n_samples)
width = np.random.uniform(2, 10, n_samples)

weight = thickness * length * width * 0.1
strength = 1000 / (thickness + 0.5) + width * 5

X = np.column_stack((thickness, length, width))
y = np.column_stack((weight, strength))

model = RandomForestRegressor()
model.fit(X, y)

os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/model.pkl")

print("Model trained and saved to models/model.pkl")
