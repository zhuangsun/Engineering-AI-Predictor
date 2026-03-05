import os
from fastapi import FastAPI
from fastapi.responses import FileResponse
from app.schemas import DesignInput, OptimizationBounds
from app.services import predict_design
from app.optimizer import run_optimization, run_pareto_optimization

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

app = FastAPI(
    title="Engineering AI Optimizer",
    description="AI-powered multi-objective design optimization platform",
    version="1.0.0"
)


@app.get("/", include_in_schema=False)
def serve_ui():
    return FileResponse(os.path.join(BASE_DIR, "templates", "index.html"))


@app.post("/predict")
def predict(data: DesignInput):
    return predict_design(data)


@app.get("/optimize")
def optimize():
    return run_optimization()


@app.post("/optimize_multi")
def optimize_multi(bounds: OptimizationBounds):
    return run_pareto_optimization(bounds.model_dump())
