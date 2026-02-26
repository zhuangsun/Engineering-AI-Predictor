from pydantic import BaseModel
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi import Request
import joblib
import numpy as np
import os

from app.optimizer import genetic_optimize

app = FastAPI()

templates = Jinja2Templates(directory="templates")

MODEL_PATH = "models/model.pkl"

if not os.path.exists(MODEL_PATH):
    raise Exception("Model not found. Run train.py first.")

model = joblib.load(MODEL_PATH)


@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/predict")
async def predict(data: dict):

    inputs = np.array([[
        data["x1"],
        data["x2"],
        data["x3"],
        data["x4"],
        data["x5"]
    ]])

    prediction = model.predict(inputs)

    return {"prediction": float(prediction[0])}


class Bounds(BaseModel):
    x1_min: float
    x1_max: float
    x2_min: float
    x2_max: float
    x3_min: float
    x3_max: float
    x4_min: float
    x4_max: float
    x5_min: float
    x5_max: float


@app.post("/optimize")
def run_optimization(bounds: Bounds):

    search_bounds = [
        (bounds.x1_min, bounds.x1_max),
        (bounds.x2_min, bounds.x2_max),
        (bounds.x3_min, bounds.x3_max),
        (bounds.x4_min, bounds.x4_max),
        (bounds.x5_min, bounds.x5_max),
    ]

    params, value = genetic_optimize(search_bounds)

    return {
        "best_parameters": params.tolist(),
        "max_prediction": float(value)
    }
