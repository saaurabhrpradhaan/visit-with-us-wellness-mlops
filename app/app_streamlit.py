"""
Wellness Tourism Package Purchase Predictor - Streamlit App
Deployed on Hugging Face Spaces
"""

import streamlit as st
import pandas as pd
import skops.io as sio
from huggingface_hub import hf_hub_download
import numpy as np

# Load model from HF Hub
@st.cache_resource
def load_model():
    model_file = hf_hub_download(
        repo_id="SaaurabhR/wellness-wtp-rf-model",
        filename="wellness_rf.skops",
        repo_type="model",
    )
    return sio.load(model_file)

st.title("🧳 Wellness Tourism Package Predictor")
st.markdown("Predict if a customer will purchase the Wellness Tourism Package")

model = load_model()

# Input form for all features
with st.form("customer_form"):
    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("Age", min_value=18, max_value=80, value=35)
        gender = st.selectbox("Gender", ["Male", "Female"])
        marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
        occupation = st.selectbox("Occupation", ["Salaried", "Employee", "Freelancer", "Small Business"])

    with col2:
        city_tier = st.selectbox("City Tier", [1, 2, 3])
        typeofcontact = st.selectbox("Type of Contact", ["Company Invited", "Self Inquiry"])
        passport = st.selectbox("Has Passport", [0, 1])
        own_car = st.selectbox("Owns Car", [0, 1])

    col3, col4 = st.columns(2)
    with col3:
        num_person_visiting = st.number_input("Number of People Visiting", min_value=1, max_value=10, value=2)
        num_children_visiting = st.number_input("Number of Children (<5 yrs)", min_value=0, max_value=5, value=0)
        num_trips = st.number_input("Annual Trips", min_value=0, max_value=20, value=2)

    with col4:
        monthly_income = st.number_input("Monthly Income", min_value=10000, max_value=1000000, value=50000)
        preferred_property_star = st.selectbox("Preferred Hotel Stars", [1, 2, 3, 4, 5])
        pitch_satisfaction_score = st.slider("Pitch Satisfaction Score", 0, 100, 50)

    duration_of_pitch = st.slider("Pitch Duration (minutes)", 1, 30, 10)
    number_of_followups = st.slider("Number of Follow-ups", 0, 10, 2)

    submit = st.form_submit_button("Predict Purchase Probability")

if submit:
    # Build input dataframe
    input_data = {
        "Age": [age],
        "TypeofContact": [typeofcontact],
        "CityTier": [city_tier],
        "Occupation": [occupation],
        "Gender": [gender],
        "NumberOfPersonVisiting": [num_person_visiting],
        "PreferredPropertyStar": [preferred_property_star],
        "MaritalStatus": [marital_status],
        "NumberOfTrips": [num_trips],
        "Passport": [passport],
        "OwnCar": [own_car],
        "NumberOfChildrenVisiting": [num_children_visiting],
        "PitchSatisfactionScore": [pitch_satisfaction_score],
        "MonthlyIncome": [monthly_income],
        "DurationOfPitch": [duration_of_pitch],
        "NumberOfFollowups": [number_of_followups],
        "ChildrenRatio": [num_children_visiting / max(num_person_visiting, 1)],
        "IncomePerPerson": [monthly_income / max(num_person_visiting, 1)],
    }

    X_input = pd.DataFrame(input_data)

    # Predict using the dataframe
    proba = model.predict_proba(X_input)[0, 1]
    prediction = "Will Purchase" if proba >= 0.5 else "Will Not Purchase"

    # Results
    st.success(f"**Purchase Probability: {proba:.2%}**")
    st.balloons()

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Prediction", prediction)
    with col2:
        st.metric("Confidence", f"{max(proba, 1 - proba):.1%}")
