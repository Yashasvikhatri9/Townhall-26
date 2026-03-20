import streamlit as st
import joblib
import numpy as np

# Load models
lin_model = joblib.load("titanic_linear_model.pkl")
log_model = joblib.load("titanic_logistic_model.pkl")

st.title("Titanic Survival Prediction")

# Inputs
pclass = st.number_input("Pclass", 1, 3, 1)
age = st.number_input("Age", 0, 100, 25)
sibsp = st.number_input("SibSp", 0, 10, 0)
parch = st.number_input("Parch", 0, 10, 0)
fare = st.number_input("Fare", 0.0, 500.0, 50.0)

# Input array
input_data = np.array([[pclass, age, sibsp, parch, fare]])

# Logistic Prediction
if st.button("Predict (Logistic)"):
    pred = log_model.predict(input_data)[0]
    prob = log_model.predict_proba(input_data)[0][1]

    st.write("Prediction:", "Survived" if pred == 1 else "Not Survived")
    st.write("Probability:", prob)

# Linear Prediction
if st.button("Predict (Linear)"):
    pred_cont = lin_model.predict(input_data)[0]
    pred = 1 if pred_cont >= 0.5 else 0

    st.write("Prediction:", "Survived" if pred == 1 else "Not Survived")
    st.write("Raw Output:", pred_cont)