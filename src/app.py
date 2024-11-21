
import streamlit as st
import pandas as pd
import pickle


# Load the trained model
with open(r'C:\Users\Dell\Streamlit-ML-App_Classification\model\churn_model.pkl', 'rb') as file:
    model = pickle.load(file)

# Streamlit App Title
st.title("Customer Churn Prediction App")
st.write("Enter the customer details below to predict churn.")

# Input Features
st.sidebar.header("Customer Information")
gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
senior_citizen = st.sidebar.selectbox("Senior Citizen", ["Yes", "No"])
partner = st.sidebar.selectbox("Has Partner?", ["Yes", "No"])
dependents = st.sidebar.selectbox("Has Dependents?", ["Yes", "No"])
tenure = st.sidebar.slider("Tenure (Months)", 0, 72, 12)
phone_service = st.sidebar.selectbox("Phone Service", ["Yes", "No"])
multiple_lines = st.sidebar.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])
internet_service = st.sidebar.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
online_security = st.sidebar.selectbox("Online Security", ["Yes", "No", "No internet service"])
online_backup = st.sidebar.selectbox("Online Backup", ["Yes", "No", "No internet service"])
device_protection = st.sidebar.selectbox("Device Protection", ["Yes", "No", "No internet service"])
tech_support = st.sidebar.selectbox("Tech Support", ["Yes", "No", "No internet service"])
streaming_tv = st.sidebar.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
streaming_movies = st.sidebar.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])
contract = st.sidebar.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
paperless_billing = st.sidebar.selectbox("Paperless Billing", ["Yes", "No"])
payment_method = st.sidebar.selectbox("Payment Method", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])
monthly_charges = st.sidebar.number_input("Monthly Charges", 0.0, 200.0, 50.0)
total_charges = st.sidebar.number_input("Total Charges", 0.0, 10000.0, 500.0)

# Create a dictionary of inputs
input_features = {
    'Gender': gender,
    'SeniorCitizen': 1 if senior_citizen == "Yes" else 0,
    'Partner': 1 if partner == "Yes" else 0,
    'Dependents': 1 if dependents == "Yes" else 0,
    'Tenure': tenure,
    'PhoneService': 1 if phone_service == "Yes" else 0,
    'MultipleLines': multiple_lines,
    'InternetService': internet_service,
    'OnlineSecurity': online_security,
    'OnlineBackup': online_backup,
    'DeviceProtection': device_protection,
    'TechSupport': tech_support,
    'StreamingTV': streaming_tv,
    'StreamingMovies': streaming_movies,
    'Contract': contract,
    'PaperlessBilling': 1 if paperless_billing == "Yes" else 0,
    'PaymentMethod': payment_method,
    'MonthlyCharges': monthly_charges,
    'TotalCharges': total_charges
}

# Convert inputs to DataFrame (match the format used during training)
input_df = pd.DataFrame([input_features])

# Feature encoding (if necessary)
# Example: Convert categorical features to numerical values based on your training data
# This part will vary based on how you preprocessed the data during training
input_df = pd.get_dummies(input_df, drop_first=True)

# Predict churn
if st.button("Predict Churn"):
    prediction = model.predict(input_df)
    if prediction[0] == 1:
        st.error("This customer is likely to churn.")
    else:
        st.success("This customer is not likely to churn.")
