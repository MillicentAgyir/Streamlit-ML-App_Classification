import streamlit as st
import pandas as pd
import os
import datetime
import pickle

# Load the trained model
@st.cache_resource
def load_model():
    try:
        with open(r'C:\Users\Dell\Streamlit-ML-App_Classification\model\churn_model.pkl', 'rb') as file:
            model = pickle.load(file)
        return model
    except FileNotFoundError:
        st.error("Model file not found. Please make sure the model is available at the specified path.")
        return None


model = load_model()

# Authentication Function
def authenticate():
    st.title("Authentication")
    st.write("Please log in to access the application.")
    username = st.text_input("Username", key="auth_username")
    password = st.text_input("Password", type="password", key="auth_password")

    if st.button("Login"):
        if username == "admin" and password == "password123":
            st.session_state["authenticated"] = True
            st.session_state["navigate_to_login"] = False
            st.success("Login successful! Redirecting...")
            st.rerun()
        else:
            st.error("Invalid credentials. Please try again.")


# Sidebar Navigation
def navigation():
    st.sidebar.title("Navigation")
    return st.sidebar.radio("Go to", ["Home", "View Data", "Dashboard", "Predict", "History"])


# Homepage
def homepage():
    st.title("Welcome to the Attrition Insight Application")
    st.subheader("Key Features")
    st.markdown("""
    - View key metrics and patterns from your dataset.
    - Make customer churn predictions in real time.
    - Accessible and user-friendly interface for business insights.
    """)
    st.subheader("How to run the application")
    st.markdown("""
    1. Log in using your credentials.
    2. Navigate between pages to view dashboards, predictions, and history.
    """)
    st.subheader("Need Help?")
    st.markdown("For assistance, email us at: **support@attritioninsight.com**")

    if st.button("Explore the App"):
        st.session_state["navigate_to_login"] = True
        st.rerun()


# View Data Page
def view_data_page():
    st.title("View Data")
    st.subheader("View Proprietary Data from Vodafone")
    try:
        if os.path.exists("data.csv"):
            data = pd.read_csv("data.csv")  # Replace with your dataset path
            st.dataframe(data)
        else:
            st.warning("Data file not found. Please upload or add your dataset.")
    except Exception as e:
        st.error(f"An error occurred: {e}")


# Dashboard Page (Placeholder)
def dashboard_page():
    st.title("Dashboard")
    st.write("The dashboard will display here when the dataset is added. Replace this placeholder when ready.")


# Predict Page
def predict_page():
    st.title("Predict Customer Churn")

    if model is None:
        st.error("The prediction model is not loaded. Please check the model file.")
        return

    # Input fields for prediction
    st.subheader("Personal Info 🧑‍💼")
    gender = st.selectbox("Gender", ["Male", "Female"])
    senior_citizen = st.selectbox("Senior Citizen", ["Yes", "No"])
    partner = st.selectbox("Partner", ["Yes", "No"])
    dependents = st.selectbox("Dependents", ["Yes", "No"])

    st.subheader("Work Info 💼")
    tenure = st.slider("Tenure (Months)", 0, 72, 12)
    phone_service = st.selectbox("Phone Service", ["Yes", "No"])
    multiple_lines = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])
    internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    online_security = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
    online_backup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
    device_protection = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])
    tech_support = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
    streaming_tv = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
    streaming_movies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])
    contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])
    payment_method = st.selectbox(
        "Payment Method",
        [
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)",
        ],
    )
    monthly_charges = st.number_input("Monthly Charges", 0.0, 200.0, 50.0)
    total_charges = st.number_input("Total Charges", 0.0, 10000.0, 500.0)

    input_features = {
        "Gender": gender,
        "SeniorCitizen": 1 if senior_citizen == "Yes" else 0,
        "Partner": 1 if partner == "Yes" else 0,
        "Dependents": 1 if dependents == "Yes" else 0,
        "Tenure": tenure,
        "PhoneService": 1 if phone_service == "Yes" else 0,
        "MultipleLines": multiple_lines,
        "InternetService": internet_service,
        "OnlineSecurity": online_security,
        "OnlineBackup": online_backup,
        "DeviceProtection": device_protection,
        "TechSupport": tech_support,
        "StreamingTV": streaming_tv,
        "StreamingMovies": streaming_movies,
        "Contract": contract,
        "PaperlessBilling": 1 if paperless_billing == "Yes" else 0,
        "PaymentMethod": payment_method,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges,
    }
    input_df = pd.DataFrame([input_features])

    @st.cache_data
    def preprocess_input(input_df, trained_features):
        input_df = pd.get_dummies(input_df, drop_first=True)
        missing_features = [feature for feature in trained_features if feature not in input_df.columns]
        if missing_features:
            missing_df = pd.DataFrame(0, index=input_df.index, columns=missing_features)
            input_df = pd.concat([input_df, missing_df], axis=1)
        input_df = input_df[trained_features]
        return input_df

    trained_features = model.feature_names_in_
    input_df = preprocess_input(input_df, trained_features)

    if st.button("Predict"):
        prediction = model.predict(input_df)[0]
        st.success(f"The customer is likely to churn: {'Yes' if prediction == 1 else 'No'}")


# History Page
def history_page():
    st.title("Prediction History")
    try:
        history = pd.read_csv("history.csv")
        st.dataframe(history)
    except FileNotFoundError:
        st.error("No history found. Please make some predictions first!")
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")


# Main App
def main_app():
    selected_page = navigation()

    if selected_page == "Home":
        homepage()
    elif selected_page == "View Data":
        if st.session_state.get("authenticated", False):
            view_data_page()
        else:
            st.warning("You must log in first.")
            authenticate()
    elif selected_page == "Dashboard":
        if st.session_state.get("authenticated", False):
            dashboard_page()
        else:
            st.warning("You must log in first.")
            authenticate()
    elif selected_page == "Predict":
        if st.session_state.get("authenticated", False):
            predict_page()
        else:
            st.warning("You must log in first.")
            authenticate()
    elif selected_page == "History":
        if st.session_state.get("authenticated", False):
            history_page()
        else:
            st.warning("You must log in first.")
            authenticate()


# Application Entry Point
if "navigate_to_login" in st.session_state and st.session_state["navigate_to_login"]:
    authenticate()
else:
    main_app()
