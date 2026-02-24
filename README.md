# Engineering AI Predictor

## Overview

Engineering AI Predictor is a lightweight AI-driven system that simulates an engineering parameter prediction workflow.

The system demonstrates:

- Synthetic engineering data generation
- Machine learning model training
- REST API deployment
- Interactive web interface

---

## System Architecture

Data Generation → Model Training → Model Serialization → REST API → Streamlit UI

---

## Tech Stack

- Python 3.9
- Scikit-learn
- FastAPI
- Streamlit
- Uvicorn
- Git

---

## How to Run

### 1. Train Model

python app/train.py


### 2. Start API

uvicorn app.api:app --reload


### 3. Start UI

streamlit run streamlit_app.py


---

## Future Improvements

- Real engineering dataset integration
- Hyperparameter tuning
- Optimization module
- Docker deployment
- Cloud hosting
