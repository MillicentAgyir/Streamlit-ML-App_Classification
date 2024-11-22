import streamlit as st
import pandas as pd
import os
import datetime
import pickle

# Load the trained model
@st.cache_resource
def load_model():
    with open(r'C:\Users\Dell\Streamlit-ML-App_Classification\model\churn_model.pkl', 'rb') as file:
        model = pickle.load(file)
    return model


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
        else:
            st.error("Invalid credentials. Please try again.")


# Sidebar Navigation
def navigation():
    st.sidebar.title("Navigation")
    return st.sidebar.radio("Go to", ["Dashboard", "Predict", "History"])


# Dashboard Page (Placeholder)
def dashboard_page():
    st.title("Dashboard")
    st.write("The dashboard will display here when the dataset is added. Replace this placeholder when ready.")


# Predict Page
def predict_page():
    st.title("Predict Customer Churn")

    # Create Sections for Inputs
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

    # Input Features
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

    # Preprocess Input Data
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

    # Predict Button
    if st.button("Predict"):
        prediction = model.predict(input_df)[0]
        serializable_features = {
            k: int(v) if isinstance(v, bool) else v for k, v in input_features.items()
        }
        serializable_features["Timestamp"] = str(datetime.datetime.now())
        serializable_features["Prediction"] = "Yes" if prediction == 1 else "No"

        if os.path.exists("history.csv"):
            history = pd.read_csv("history.csv")
        else:
            history = pd.DataFrame(columns=list(serializable_features.keys()))

        new_record = pd.DataFrame([serializable_features])
        history = pd.concat([history, new_record], ignore_index=True)
        history.to_csv("history.csv", index=False)

        # Display Prediction Result
        if prediction == 1:
            st.error("This customer is likely to churn.")
        else:
            st.success("This customer is not likely to churn.")


# History Page
def history_page():
    st.title("Prediction History")
    try:
        history = pd.read_csv("history.csv")
        st.dataframe(history)
    except FileNotFoundError:
        st.error("No history found. Make some predictions first!")
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")


# Main Application
if "authenticated" not in st.session_state:
    st.session_state["authenticated"] = False

if not st.session_state["authenticated"]:
    authenticate()
else:
    selected_page = navigation()

    if selected_page == "Dashboard":
        dashboard_page()
    elif selected_page == "Predict":
        predict_page()
    elif selected_page == "History":
        history_page()
