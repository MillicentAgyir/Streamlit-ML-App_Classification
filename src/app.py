
import streamlit as st
import pandas as pd
import os
import datetime
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px


# Load the trained model
@st.cache_resource
def load_model():
    with open(r'C:\Users\Dell\Streamlit-ML-App_Classification\model\churn_model.pkl', 'rb') as file:
        model = pickle.load(file)
    return model


model = load_model()

# Sidebar Navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Predict", "History"])

# Predict Page
if page == "Predict":
    # Page Title
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
        # Apply one-hot encoding
        input_df = pd.get_dummies(input_df, drop_first=True)

        # Identify missing features
        missing_features = [feature for feature in trained_features if feature not in input_df.columns]

         # Add all missing features at once
        if missing_features:
        # Create a DataFrame with missing features, filled with zeros
           missing_df = pd.DataFrame(0, index=input_df.index, columns=missing_features)
           input_df = pd.concat([input_df, missing_df], axis=1)

    # Ensure column order matches the model's training data
        input_df = input_df[trained_features]

        return input_df
    
    # Get the feature names from the trained model
    trained_features = model.feature_names_in_

    # Preprocess input data
    input_df = preprocess_input(input_df, trained_features)

    # Predict Button
    if st.button("Predict"):
        # Make the prediction
        prediction = model.predict(input_df)[0]

        # Ensure all values in `input_features` are JSON serializable
        serializable_features = {
            k: int(v) if isinstance(v, bool) else v for k, v in input_features.items()
        }

        # Add prediction and timestamp to the input features
        serializable_features["Timestamp"] = str(datetime.datetime.now())
        serializable_features["Prediction"] = "Yes" if prediction == 1 else "No"

        # Check if the history file exists
        if os.path.exists("history.csv"):
            # Load the existing history
            history = pd.read_csv("history.csv")
        else:
            # Create an empty DataFrame with appropriate columns
            history = pd.DataFrame(columns=list(serializable_features.keys()))

        # Create a new DataFrame for the current prediction
        new_record = pd.DataFrame([serializable_features])  # <-- Define the new record here

        # Concatenate the new record to the history DataFrame
        history = pd.concat([history, new_record], ignore_index=True)

        # Save the updated history back to the CSV file
        history.to_csv("history.csv", index=False)

        # Display Prediction Result
        if prediction == 1:
            st.error("This customer is likely to churn.")
        else:
            st.success("This customer is not likely to churn.")

# History Page
elif page == "History":
    st.title("Prediction History")

    try:
        # Load the history file
        history = pd.read_csv("history.csv")

        # Display the DataFrame as a table
        st.dataframe(history)

    except FileNotFoundError:
        st.error("No history found. Make some predictions first!")
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")


   


