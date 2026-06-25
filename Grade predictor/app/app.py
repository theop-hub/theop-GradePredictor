import streamlit as st
import joblib
import os

baseDir = os.path.dirname(os.path.abspath(__file__))
modelPath = os.path.join(baseDir, "../model/model.pkl")
model = joblib.load(modelPath)

st.title("Student Score Predictor")
st.write("Enter your study habits and get predicted score.")


hours = st.slider("Hours Studied", 0, 12, 4)
sleep = st.slider("Sleep Hours", 0, 12, 7)

if st.button("Predict"):
    prediction = model.predict([[hours, sleep]])
    st.success(f"Predicted score: {prediction[0]:.2f}")
