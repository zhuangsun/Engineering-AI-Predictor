# Engineering AI Optimization Platform

A full-stack web application that uses a **Random Forest surrogate model** to replace expensive numerical simulations in structural design optimization. Users can predict performance in real time, explore the **Pareto-optimal trade-off** between weight and strength via two optimization algorithms, run single-variable sensitivity sweeps, and export results — all from an interactive browser UI.

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
│  Inputs : thickness, length, width  (mm)                        │
│  Outputs: weight (kg), strength (MPa)   R² > 0.98               │
└──────────────────────────────────────────────────────────────────┘
```

### Key modules

| File | Role |
|---|---|
| `train.py` | Generates synthetic dataset, 80/20 train/test split, prints R² & MAE, serialises model |
| `app/main.py` | FastAPI app — all routes, request-logging middleware |
| `app/services.py` | Model loading, single-point prediction with per-tree uncertainty |
| `app/optimizer.py` | Non-dominated sorting, NSGA-II (SBX + polynomial mutation), sensitivity sweep |
| `app/schemas.py` | Pydantic request models with validation |
| `templates/index.html` | Single-page UI: prediction, optimization, sensitivity, CSV export |

---

## Features

| Feature | Detail |
|---|---|
| **Instant prediction** | Returns weight & strength with `±` uncertainty (RF inter-tree std dev) |
| **Feature importance** | Horizontal bar chart loaded on page load via `GET /feature_importance` |
| **Random-sampling Pareto front** | 2 000 samples → non-dominated sorting → interactive scatter chart |
| **NSGA-II optimization** | Evolutionary algorithm with SBX crossover and polynomial mutation; configurable population size and generations |
| **Sensitivity analysis** | Sweep any one design variable while fixing the other two; dual-axis line chart |
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
| `POST` | `/predict` | Predict weight & strength with uncertainty |
| `GET` | `/feature_importance` | RF feature importances for all inputs |
| `POST` | `/optimize_multi` | Pareto front via random sampling |
| `POST` | `/optimize_ga` | Pareto front via NSGA-II |
| `POST` | `/sensitivity` | Single-variable sweep (weight & strength vs. one input) |
| `GET` | `/docs` | Swagger UI |

---

## Design Space

| Variable | Range | Unit |
|---|---|---|
| Thickness | 1 – 10 | mm |
| Length | 5 – 20 | mm |
| Width | 2 – 10 | mm |

Synthetic ground-truth functions used for training:

```
weight   = t * l * w * 0.1
strength = 1000 / (t + 0.5) + w * 5
```

These capture the physical intuition that thinner cross-sections reduce weight but compromise strength — a classic structural trade-off.

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
