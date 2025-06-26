# app.py

import streamlit as st
import numpy as np
import joblib

# Load the trained model
model = joblib.load("model.pkl")

st.title("💉 Diabetes Prediction App")

# Create input fields..............

pregnancies = st.number_input("Pregnancies", min_value=0)

glucose = st.number_input("Glucose Level", min_value=0)

blood_pressure = st.number_input("Blood Pressure", min_value=0)

skin_thickness = st.number_input("Skin Thickness", min_value=0)

insulin = st.number_input("Insulin Level", min_value=0)

bmi = st.number_input("BMI", min_value=0.0)

dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0)

age = st.number_input("Age", min_value=0)

# Predict button....................
if st.button("Predict"):

    # Convert input into NumPy array............

    input_data = np.array([[pregnancies, glucose, blood_pressure, skin_thickness,
                            insulin, bmi, dpf, age]])

    # Make prediction..............

    prediction = model.predict(input_data)

    # Show result...............

    if prediction[0] == 1:
        st.error("⚠️ You may be diabetic.")
    else:
        st.success("✅ You are not diabetic.")
