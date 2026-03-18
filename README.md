# Engineering AI Optimization Platform

A full-stack web application that uses a **Random Forest surrogate model** on the **Welded Beam** engineering benchmark (Ragsdell & Phillips, 1976). Users can predict manufacturing cost and tip deflection in real time, explore the **Pareto-optimal trade-off** between the two objectives via two optimization algorithms, run single-variable sensitivity sweeps, and export results — all from an interactive browser UI.

---

## Motivation

In numerical simulation workflows, evaluating a single design can take minutes to hours (FEA, CFD, etc.). Surrogate-model-based optimization replaces that bottleneck with a fast ML model trained on simulation data, enabling rapid multi-objective design space exploration — the same methodology used in aerospace, civil, and mechanical engineering.

---

## Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│  Browser  (templates/index.html)                                 │
│                                                                  │
│  Predict form  |  Optimizer (Sampling / NSGA-II)  |  Sensitivity │
└────────┬───────────────────┬──────────────────────┬─────────────┘
         │ POST /predict     │ POST /optimize_multi  │ POST /sensitivity
         │                   │ POST /optimize_ga     │
         ▼                   ▼                       ▼
┌──────────────────────────────────────────────────────────────────┐
│  FastAPI  (app/main.py)   +  request logging middleware          │
│                                                                  │
│  /predict            → services.predict_design()                │
│  /feature_importance → model.feature_importances_               │
│  /optimize_multi     → optimizer.run_pareto_optimization()      │
│  /optimize_ga        → optimizer.run_ga_optimization()          │
│  /sensitivity        → optimizer.run_sensitivity()              │
└───────────────────────────┬──────────────────────────────────────┘
                            │
                            ▼
┌──────────────────────────────────────────────────────────────────┐
│  scikit-learn RandomForestRegressor  (100 trees)                 │
│  models/model.pkl  —  trained by train.py (80/20 split)         │
│                                                                  │
│  Inputs : h, l, t, b  (Welded Beam design variables)            │
│  Outputs: cost ($), deflection (in)   R² = 0.99 / 0.89          │
└──────────────────────────────────────────────────────────────────┘
```

### Key modules

| File | Role |
|---|---|
| `train.py` | Generates Welded Beam dataset (feasibility-filtered), 80/20 split, prints R² & MAE, serialises model |
| `app/main.py` | FastAPI app — all routes, request-logging middleware |
| `app/services.py` | Model loading, single-point prediction with per-tree uncertainty |
| `app/optimizer.py` | Non-dominated sorting, NSGA-II (SBX + polynomial mutation), sensitivity sweep |
| `app/schemas.py` | Pydantic request models with validation |
| `templates/index.html` | Single-page UI: prediction, optimization, sensitivity, CSV export |

---

## Features

| Feature | Detail |
|---|---|
| **Instant prediction** | Returns cost & deflection with `±` uncertainty (RF inter-tree std dev) |
| **Feature importance** | Horizontal bar chart loaded on page load via `GET /feature_importance` |
| **Random-sampling Pareto front** | 2 000 samples → non-dominated sorting → interactive scatter chart |
| **NSGA-II optimization** | Evolutionary algorithm with SBX crossover and polynomial mutation; configurable population size and generations |
| **Sensitivity analysis** | Sweep any one design variable while fixing the other three; dual-axis line chart |
| **CSV export** | Download the full Pareto front table with one click |
| **REST API docs** | Swagger UI at `/docs`, ReDoc at `/redoc` |
| **CI** | GitHub Actions runs the test suite on every push |
| **Docker** | Single-command deployment via `docker-compose up` |

---

## Quick Start

### 1. Install dependencies

```bash
python -m venv venv
# macOS / Linux
source venv/bin/activate
# Windows (Git Bash)
source venv/Scripts/activate
# Windows (Command Prompt)
venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Train the surrogate model

```bash
python train.py
# Prints train/test R² and MAE, saves models/model.pkl
```

### 3. Start the server

```bash
uvicorn app.main:app --reload
```

Open [http://localhost:8000](http://localhost:8000) in your browser.
API docs at [http://localhost:8000/docs](http://localhost:8000/docs).

### Docker (alternative)

```bash
docker-compose up
```

Builds the image, trains the model, and serves the app — no manual setup needed.

---

## Running Tests

```bash
pytest tests/ -v
```

33 tests covering prediction values, Pareto mask correctness, NSGA-II structure, sensitivity monotonicity, and all API endpoints.

---

## API Reference

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Serves the frontend UI |
| `POST` | `/predict` | Predict cost & deflection with uncertainty |
| `GET` | `/feature_importance` | RF feature importances for all inputs |
| `POST` | `/optimize_multi` | Pareto front via random sampling |
| `POST` | `/optimize_ga` | Pareto front via NSGA-II |
| `POST` | `/sensitivity` | Single-variable sweep (cost & deflection vs. one input) |
| `GET` | `/docs` | Swagger UI |

---

## Design Space — Welded Beam Benchmark

A cantilever beam welded to a rigid wall, loaded with P = 6,000 lbf at the free end.

| Variable | Description | Bounds | Unit |
|---|---|---|---|
| h | Weld size | 0.1 – 2.0 | in |
| l | Weld length | 0.1 – 10.0 | in |
| t | Bar thickness | 0.1 – 10.0 | in |
| b | Bar height | 0.1 – 2.0 | in |

**Objectives (both minimised):**

```
cost       = 1.10471 · h² · l  +  0.04811 · t · b · (14 + l)   [$]
deflection = 2.1952 / (t³ · b)                                   [in]
```

**Feasibility constraints** (used to filter training data):

```
shear stress   τ  ≤ 13 600 psi
bending stress σ  ≤ 30 000 psi
deflection     δ  ≤ 0.25 in
h              ≤  b
```

These constraints mean only ~26 % of the raw random samples are feasible, giving the surrogate a genuinely non-trivial region to learn. Cost and deflection are in fundamental conflict: a stiffer beam (lower δ) requires more material, which raises cost.

**Surrogate performance** (100-tree RF, 80/20 split, ~2 600 feasible samples):

| Output | R² | MAE |
|---|---|---|
| Cost ($) | 0.986 | 0.69 |
| Deflection (in) | 0.892 | 0.0012 |

The deflection R² of 0.89 — lower than a trivial synthetic formula — reflects the genuine nonlinearity of 1/(t³·b) in the feasible region.

**Reference:** Ragsdell, K. M. & Phillips, D. T. (1976). *Optimal Design of a Class of Welded Structures Using Geometric Programming.* ASME Journal of Engineering for Industry.

---

## Tech Stack

| Layer | Technology |
|---|---|
| Backend | Python 3.10+, FastAPI, Uvicorn |
| ML | scikit-learn (RandomForestRegressor), NumPy, joblib |
| Frontend | Vanilla JS, Chart.js |
| Testing | pytest, httpx |
| DevOps | Docker, GitHub Actions CI |

---

## Project Structure

```
Engineering-AI-Predictor/
├── .github/
│   └── workflows/
│       └── test.yml         # CI: runs pytest on every push
├── app/
│   ├── main.py              # FastAPI app + logging middleware
│   ├── optimizer.py         # Pareto filter, NSGA-II, sensitivity sweep
│   ├── schemas.py           # Pydantic request models
│   └── services.py          # Model loading & prediction with uncertainty
├── templates/
│   └── index.html           # Single-page frontend
├── tests/
│   ├── test_predict.py      # Prediction + feature importance tests
│   └── test_optimizer.py    # Pareto mask, NSGA-II, sensitivity, API tests
├── models/                  # Serialised model (generated by train.py)
├── train.py                 # Training script with evaluation metrics
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── README.md
```

---

## License

MIT
