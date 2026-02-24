import streamlit as st
import requests

st.title("Engineering AI Parameter Predictor")

st.write("Enter engineering input parameters below:")

# 5 input sliders
f1 = st.slider("Parameter 1", 0.0, 1.0, 0.5)
f2 = st.slider("Parameter 2", 0.0, 1.0, 0.5)
f3 = st.slider("Parameter 3", 0.0, 1.0, 0.5)
f4 = st.slider("Parameter 4", 0.0, 1.0, 0.5)
f5 = st.slider("Parameter 5", 0.0, 1.0, 0.5)

if st.button("Predict"):

    data = {
        "features": [f1, f2, f3, f4, f5]
    }

    try:
        response = requests.post(
            "http://127.0.0.1:8000/predict",
            json=data
        )

        result = response.json()

        st.success(f"Predicted Engineering Output: {result['prediction']:.4f}")

    except Exception as e:
        st.error("API not running. Please start the backend first.")
