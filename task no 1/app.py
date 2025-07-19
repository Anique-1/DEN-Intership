import streamlit as st
import pickle
import numpy as np
import pandas as pd

#Load models
with open('decision_tree_model.pkl', 'rb') as f:
    dt_model = pickle.load(f)
with open('svm_model.pkl', 'rb') as f:
    rf_model = pickle.load(f)

st.title("Student Performance Grade Class Prediction")


model_choice = st.sidebar.selectbox("Choose Model", ("Decision Tree", "Random Forest"))

st.header("Input Student Features")


age = st.number_input("Age", min_value=10, max_value=25, value=16)
gender = st.selectbox("Gender", options=["Male", "Female"])
ethnicity = st.selectbox("Ethnicity", options=["Group A", "Group B", "Group C", "Group D", "Group E"])  # Adjust options as per your data
parental_education = st.selectbox("Parental Education", options=["High School", "Some College", "Bachelor's", "Master's", "PhD"])  # Adjust as per your data
study_time_weekly = st.number_input("Study Time Weekly (hours)", min_value=0.0, max_value=100.0, value=10.0)
absences = st.number_input("Absences", min_value=0, max_value=100, value=2)
tutoring = st.selectbox("Tutoring", options=["Yes", "No"])
parental_support = st.selectbox("Parental Support", options=["Yes", "No"])
extracurricular = st.selectbox("Extracurricular", options=["Yes", "No"])
sports = st.selectbox("Sports", options=["Yes", "No"])
music = st.selectbox("Music", options=["Yes", "No"])
volunteering = st.selectbox("Volunteering", options=["Yes", "No"])
gpa = st.number_input("GPA", min_value=0.0, max_value=4.0, value=3.0)


gender_map = {"Male": 0, "Female": 1}
tutoring_map = {"No": 0, "Yes": 1}
parental_support_map = {"No": 0, "Yes": 1}
extracurricular_map = {"No": 0, "Yes": 1}
sports_map = {"No": 0, "Yes": 1}
music_map = {"No": 0, "Yes": 1}
volunteering_map = {"No": 0, "Yes": 1}


ethnicity_map = {"Group A": 0, "Group B": 1, "Group C": 2, "Group D": 3, "Group E": 4}
parental_education_map = {"High School": 0, "Some College": 1, "Bachelor's": 2, "Master's": 3, "PhD": 4}

input_features = np.array([[age,
                           gender_map[gender],
                           ethnicity_map[ethnicity],
                           parental_education_map[parental_education],
                           study_time_weekly,
                           absences,
                           tutoring_map[tutoring],
                           parental_support_map[parental_support],
                           extracurricular_map[extracurricular],
                           sports_map[sports],
                           music_map[music],
                           volunteering_map[volunteering],
                           gpa]])

if st.button("Predict Grade Class"):
    if model_choice == "Decision Tree":
        prediction = dt_model.predict(input_features)
        st.success(f"Decision Tree Predicted Grade Class: {prediction[0]}")
    else:
        prediction = rf_model.predict(input_features)
        st.success(f"Random Forest Predicted Grade Class: {prediction[0]}") 