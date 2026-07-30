import streamlit as st
import joblib
import pandas as pd

# Load the trained model
model = joblib.load("Model/insurance_model.pkl")

st.set_page_config(
    page_title="Medical Insurance Cost Prediction",
    page_icon="🏥",
    layout="centered"
)

st.title("🏥 Medical Insurance Cost Prediction")

st.write("Enter the customer details below to predict annual medical insurance charges.")

# User Inputs
age = st.slider("Age", 18, 64, 30)

sex = st.selectbox(
    "Sex",
    ["Male", "Female"]
)

bmi = st.number_input(
    "BMI",
    min_value=10.0,
    max_value=60.0,
    value=25.0
)

children = st.slider(
    "Number of Children",
    0,
    5,
    0
)

smoker = st.selectbox(
    "Smoker",
    ["No", "Yes"]
)

region = st.selectbox(
    "Region",
    ["Northeast", "Northwest", "Southeast", "Southwest"]
)

# Feature Engineering
smoker_binary = 1 if smoker == "Yes" else 0
smoker_bmi = smoker_binary * bmi

sex_male = 1 if sex == "Male" else 0

region_northwest = 1 if region == "Northwest" else 0
region_southeast = 1 if region == "Southeast" else 0
region_southwest = 1 if region == "Southwest" else 0

bmi_category_obese = 1 if bmi >= 30 else 0
bmi_category_overweight = 1 if 25 <= bmi < 30 else 0
bmi_category_underweight = 1 if bmi < 18.5 else 0

# Create input dataframe
input_data = pd.DataFrame({
    "age":[age],
    "bmi":[bmi],
    "children":[children],
    "smoker_binary":[smoker_binary],
    "smoker_bmi":[smoker_bmi],
    "sex_male":[sex_male],
    "region_northwest":[region_northwest],
    "region_southeast":[region_southeast],
    "region_southwest":[region_southwest],
    "bmi_category_Obese":[bmi_category_obese],
    "bmi_category_Overweight":[bmi_category_overweight],
    "bmi_category_Underweight":[bmi_category_underweight]
})

# Prediction
if st.button("Predict Insurance Charges"):
    prediction = model.predict(input_data)[0]

    st.success(f"Estimated Insurance Charges: ₹ {prediction:,.2f}")