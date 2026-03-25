import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler

# --- 1. SET UP PAGE CONFIG ---
st.set_page_config(page_title="HR Attrition Predictor", layout="wide")

st.title("📊 HR Employee Attrition Prediction")
st.markdown("""
Predict whether an employee is likely to leave the company based on demographic and job-related factors.
""")

# --- 2. DATA PREPROCESSING (Matching your Notebook) ---
# In a real app, you would load your saved .pkl model. 
# Here, we simulate the logic found in your 'hr_attrition_prediction.ipynb'.
def preprocess_and_train_mock():
    # This represents the columns identified in your notebook
    cat_cols = ['BusinessTravel', 'Department', 'EducationField', 'Gender', 'JobRole', 'MaritalStatus', 'OverTime']
    # Note: Your notebook dropped 'EmployeeCount', 'EmployeeNumber', 'Over18', 'StandardHours'
    return cat_cols

# --- 3. SIDEBAR - USER INPUTS ---
st.sidebar.header("Employee Attributes")

def user_input_features():
    # Numerical Inputs based on your num_cols list
    age = st.sidebar.slider("Age", 18, 60, 30)
    monthly_income = st.sidebar.number_input("Monthly Income ($)", 1000, 20000, 5000)
    total_working_years = st.sidebar.slider("Total Working Years", 0, 40, 10)
    job_satisfaction = st.sidebar.select_slider("Job Satisfaction", options=[1, 2, 3, 4])
    work_life_balance = st.sidebar.select_slider("Work-Life Balance", options=[1, 2, 3, 4])
    
    # Categorical Inputs based on your cat_cols list
    overtime = st.sidebar.selectbox("Does the employee work Overtime?", ["Yes", "No"])
    job_role = st.sidebar.selectbox("Job Role", [
        "Sales Executive", "Research Scientist", "Laboratory Technician", 
        "Manufacturing Director", "Healthcare Representative", "Manager", 
        "Sales Representative", "Research Director", "Human Resources"
    ])
    biz_travel = st.sidebar.selectbox("Business Travel", ["Travel_Rarely", "Travel_Frequently", "Non-Travel"])
    
    data = {
        'Age': age,
        'MonthlyIncome': monthly_income,
        'TotalWorkingYears': total_working_years,
        'JobSatisfaction': job_satisfaction,
        'WorkLifeBalance': work_life_balance,
        'OverTime': overtime,
        'JobRole': job_role,
        'BusinessTravel': biz_travel
    }
    return pd.DataFrame(data, index=[0])

input_df = user_input_features()

# --- 4. MAIN PANEL - DISPLAY & PREDICTION ---
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Selected Parameters")
    st.write(input_df)

with col2:
    st.subheader("Prediction Analysis")
    
    # Business Logic from your Insights:
    # Overtime, Income, and Satisfaction were your key drivers
    risk_score = 0
    if input_df['OverTime'][0] == "Yes":
        risk_score += 40
        st.warning("⚠️ Overtime is a high-risk factor for this employee.")
    
    if input_df['JobSatisfaction'][0] <= 2:
        risk_score += 30
        st.info("ℹ️ Low job satisfaction increases attrition probability.")
        
    if input_df['MonthlyIncome'][0] < 4000:
        risk_score += 20

    # Display Result
    if risk_score > 50:
        st.error(f"Prediction: High Risk of Attrition (Score: {risk_score}%)")
    else:
        st.success(f"Prediction: Low Risk of Attrition (Score: {risk_score}%)")

# --- 5. INTERACTIVE INSIGHTS SECTION ---
st.divider()
st.subheader("💡 Key Insights (From your Model)")
st.info("""
- **Overtime:** Significantly increases attrition.
- **Job Satisfaction:** Low satisfaction is a strong predictor of leaving.
- **Work-Life Balance:** Poor balance is a major contributor.
- **Tenure:** Shorter tenure employees show higher attrition.
""")