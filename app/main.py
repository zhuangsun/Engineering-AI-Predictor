from fastapi import FastAPI
from app.schemas import DesignInput
from app.services import predict_design
from app.optimizer import run_optimization

app = FastAPI(
    title="Engineering AI Optimizer",
    description="AI-powered multi-objective design optimization platform",
    version="1.0.0"
)


@app.get("/")
def root():
    return {"message": "Engineering AI Optimizer is running."}


@app.post("/predict")
def predict(data: DesignInput):
    return predict_design(data)


@app.get("/optimize")
def optimize():
    return run_optimization()
