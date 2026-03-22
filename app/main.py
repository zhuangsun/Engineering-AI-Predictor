import os
import json
import time
import logging
from fastapi import FastAPI, Request
from fastapi.responses import FileResponse
from app.schemas import DesignInput, OptimizationBounds, GAOptimizationRequest, SensitivityRequest
from app.services import predict_design, model
from app.optimizer import (
    run_optimization,
    run_pareto_optimization,
    run_ga_optimization,
    run_sensitivity,
    check_feasibility,
)

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("engineering_ai")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Engineering AI Optimizer",
    description="AI-powered multi-objective design optimisation platform",
    version="2.0.0",
)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    t0 = time.perf_counter()
    response = await call_next(request)
    ms = (time.perf_counter() - t0) * 1000
    logger.info("%s %s → %d  (%.1f ms)", request.method, request.url.path, response.status_code, ms)
    return response


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/", include_in_schema=False)
def serve_ui():
    return FileResponse(os.path.join(BASE_DIR, "templates", "index.html"))


@app.post("/predict")
def predict(data: DesignInput):
    return predict_design(data)


@app.get("/feature_importance")
def feature_importance():
    """Global feature importances from the Random Forest (averaged across outputs)."""
    return {
        "features":    ["h (weld size)", "l (weld length)", "t (bar thickness)", "b (bar height)"],
        "importances": model.feature_importances_.tolist(),
    }


@app.get("/optimize")
def optimize():
    return run_optimization()


@app.post("/optimize_multi")
def optimize_multi(bounds: OptimizationBounds):
    """Pareto front via random sampling (fast)."""
    return run_pareto_optimization(bounds.model_dump())


@app.post("/optimize_ga")
def optimize_ga(req: GAOptimizationRequest):
    """Pareto front via NSGA-II evolutionary algorithm (thorough)."""
    return run_ga_optimization(req.model_dump(), req.pop_size, req.n_generations)


@app.post("/sensitivity")
def sensitivity(req: SensitivityRequest):
    """Single-variable sweep: cost and deflection as a function of one design variable."""
    return run_sensitivity(
        variable=req.variable,
        fixed_h=req.fixed_h,
        fixed_l=req.fixed_l,
        fixed_t=req.fixed_t,
        fixed_b=req.fixed_b,
        sweep_min=req.sweep_min,
        sweep_max=req.sweep_max,
        n_points=req.n_points,
    )


@app.post("/check_feasibility")
def check_feasibility_endpoint(data: DesignInput):
    """
    Check whether a design satisfies all Welded Beam structural constraints.
    Returns each constraint value, its limit, and pass/fail status.
    """
    return check_feasibility(data.h, data.l, data.t, data.b)


@app.get("/model_info")
def model_info():
    """Surrogate model metadata: training stats, R², MAE, and configuration."""
    info_path = os.path.join(BASE_DIR, "models", "model_info.json")
    if not os.path.exists(info_path):
        return {"error": "model_info.json not found — run train.py to generate it"}
    with open(info_path) as f:
        return json.load(f)
