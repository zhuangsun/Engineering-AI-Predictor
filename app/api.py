from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
import os

app = FastAPI()

# -------------------------
# Load Trained Model
# -------------------------

MODEL_PATH = "models/model.pkl"

if not os.path.exists(MODEL_PATH):
    raise Exception("Model not found. Please run train.py first.")

model = joblib.load(MODEL_PATH)


# -------------------------
# Define Input Schema
# -------------------------

class InputData(BaseModel):
    features: list[float]


# -------------------------
# Root Endpoint
# -------------------------

@app.get("/")
def root():
    return {"message": "Engineering AI Predictor API is running."}


# -------------------------
# Prediction Endpoint
# -------------------------

@app.post("/predict")
def predict(data: InputData):

    features = np.array(data.features).reshape(1, -1)

    prediction = model.predict(features)

    return {
        "prediction": float(prediction[0])
    }
