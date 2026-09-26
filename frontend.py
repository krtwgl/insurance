import streamlit as st
import requests
import os

API_HOST = os.getenv("BACKEND_URL", "http://127.0.0.1:8000")
API_URL = f"{API_HOST}/predict"
# Adjust URL if testing locally (e.g., "http://127.0.0.1:8000/predict")
#API_URL = "http://127.0.0.1:8000/predict"

st.set_page_config(page_title="Insurance Premium Predictor", layout="centered")
st.title("Insurance Premium Category Predictor")
st.markdown("Enter your details below:")

# Input fields
age = st.number_input("Age", min_value=1, max_value=119, value=30)
weight = st.number_input("Weight (kg)", min_value=1.0, value=65.0)
# Backend requires height < 2.5 meters
height = st.number_input("Height (meters)", min_value=0.5, max_value=2.49, value=1.70, step=0.01)
income_lpa = st.number_input("Annual Income (LPA)", min_value=0.1, value=10.0, step=0.5)

smoker_display = st.selectbox("Are you a smoker?", options=["No", "Yes"])
smoker_bool = True if smoker_display == "Yes" else False

city = st.text_input("City", value="Mumbai").strip()
occupation = st.selectbox(
    "Occupation",
    [
        "retired",
        "freelancer",
        "student",
        "government_job",
        "business_owner",
        "unemployed",
        "private_job",
    ],
)

if st.button("Predict Premium Category"):
    input_data = {
        "age": int(age),
        "weight": float(weight),
        "height": float(height),
        "income_lpa": float(income_lpa),
        "smoker": smoker_bool,
        "city": city,
        "occupation": occupation,
    }

    try:
        response = requests.post(API_URL, json=input_data, timeout=10)
        
        if response.status_code == 200:
            result = response.json()
            predicted_category = result.get("predicted_category", "Unknown")
            st.success(f"Predicted Insurance Premium Category: **{predicted_category}**")
        else:
            st.error(f"API Error (Status Code {response.status_code})")
            st.json(response.json())

    except requests.exceptions.ConnectionError:
        st.error("❌ Could not connect to the FastAPI server. Make sure the backend is active and reachable.")
    except requests.exceptions.Timeout:
        st.error("❌ Request timed out. Backend took too long to respond.")