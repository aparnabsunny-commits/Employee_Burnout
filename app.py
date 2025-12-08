import streamlit as st
import pickle
import numpy as np

# Load model and scaler
model = pickle.load(open("knn_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

# Sidebar inputs
st.sidebar.header("Employee Details Input")
age = st.sidebar.number_input("Age", 18, 65, 30)
gender = st.sidebar.selectbox("Gender (0=Female, 1=Male)", [0, 1])

departments = {
    "IT": 0,
    "Sales": 1,
    "HR": 2,
    "Finance": 3,
    "Operations": 4,
    "Marketing": 5,
    "Admin": 6
}

# Streamlit dropdown with names
department_name = st.sidebar.selectbox("Department", list(departments.keys()))

# Convert selected name to numeric for model
department = departments[department_name]
job_level = st.sidebar.number_input("Job Level", 1, 5, 2)
workload = st.sidebar.number_input("Workload", 0, 10, 5)
overtime_hours = st.sidebar.number_input("Overtime Hours", 0, 50, 5)
job_satisfaction = st.sidebar.number_input("Job Satisfaction (1–10)", 1, 10, 6)
stress_level = st.sidebar.number_input("Stress Level (1–10)", 1, 10, 7)
sleep_quality = st.sidebar.number_input("Sleep Quality (1–10)", 1, 10, 6)
work_life_balance = st.sidebar.number_input("Work-Life Balance (1–10)", 1, 10, 5)

# Prepare input
input_data = np.array([[age, gender, department, job_level, workload,
                        overtime_hours, job_satisfaction, stress_level,
                        sleep_quality, work_life_balance]])
input_scaled = scaler.transform(input_data)

# Title
st.title("💼 Employee Burnout Prediction")
st.write("Enter employee details in the sidebar and click Predict:")

# Prediction button
if st.button("Predict Burnout Risk"):
    # Get probability of class 1 (high risk)
    risk_prob = model.predict_proba(input_scaled)[0][1]  # probability of 1
    risk_percentage = int(risk_prob * 100)  # convert to 0–100%
    
    st.subheader("Burnout Risk Level:")
    st.progress(risk_percentage)
    
    if risk_percentage >= 50:
        st.error(f"⚠️ High Burnout Risk ({risk_percentage}%)")
        st.write("⚠️ Consider reducing workload, managing stress, and improving work-life balance.")
    else:
        st.success(f"✅ Low Burnout Risk ({risk_percentage}%)")
        st.write("✅ Employee is at low risk. Keep supporting their well-being!")