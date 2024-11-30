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
    try:
        with open(r'C:\Users\Dell\Streamlit-ML-App_Classification\model\churn_model.pkl', 'rb') as file:
            model = pickle.load(file)
        return model
    except FileNotFoundError:
        st.error("Model file not found. Please make sure the model is available at the specified path.")
        return None

model = load_model()

# Function to load and clean the dataset
def load_and_clean_data(data_path):
    try:
        if os.path.exists(data_path):
            data = pd.read_csv(data_path)
            if "CustomerID" in data.columns:
                # Remove the row containing '7590-VHVEG'
                data = data[data["CustomerID"] != "7590-VHVEG"]
            if "Churn" in data.columns:
                # Replace True/False with Yes/No
                data["Churn"] = data["Churn"].replace({True: "Yes", False: "No"})
            return data
        else:
            st.error(f"Dataset not found at {data_path}.")
            return None
    except Exception as e:
        st.error(f"An error occurred while loading the dataset: {e}")
        return None

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
    st.subheader("Proprietary Data from Vodafone")

    # Display Cleaned Combined Dataset
    st.subheader("Training Dataset")
    try:
        cleaned_combined_path = r"C:\Users\Dell\Streamlit-ML-App_Classification\Streamlit-ML-App_Classification\Datasets\cleaned_combined_dataset.csv"
        cleaned_combined_data = load_and_clean_data(cleaned_combined_path)
        if cleaned_combined_data is not None:
            st.dataframe(cleaned_combined_data)
    except Exception as e:
        st.error(f"An error occurred while loading the cleaned combined dataset: {e}")
    
    # Display Test Dataset
    st.subheader("Test Dataset")
    try:
        test_dataset_path = r"C:\Users\Dell\Streamlit-ML-App_Classification\Streamlit-ML-App_Classification\Datasets\TestData.csv"
        if os.path.exists(test_dataset_path):
            test_dataset = pd.read_csv(test_dataset_path, delimiter=";")
            st.dataframe(test_dataset)
        else:
            st.warning("Test dataset not found. Please upload or save the file at the specified path.")
    except Exception as e:
        st.error(f"An error occurred while loading the test dataset: {e}")
#Dashboard
def dashboard_page():
    st.title("Dashboard")
    dashboard_type = st.selectbox("Select Dashboard Type", ["EDA", "Analytics"])

    try:
        # Path to the dataset
        data_path = r"C:\Users\Dell\Streamlit-ML-App_Classification\Streamlit-ML-App_Classification\Datasets\cleaned_combined_dataset.csv"
        data = load_and_clean_data(data_path)

        if data is None:
            return

        
        if dashboard_type == "EDA":
            st.subheader("Exploratory Data Analysis")
            col1, col2 = st.columns(2)

            with col1:
                st.write("### Gender Distribution")
                gender_counts = data["gender"].value_counts()
                gender_pie = px.pie(
                    gender_counts,
                    values=gender_counts.values,
                    names=gender_counts.index,
                    color_discrete_sequence=px.colors.sequential.RdBu,
                )
                st.plotly_chart(gender_pie, use_container_width=True)
            
            with col2:
                st.write("### Outliers in Numerical Data")
                fig, ax = plt.subplots(figsize=(10, 8))
                sns.boxplot(data=data.select_dtypes(include=["number"]), ax=ax, palette="Set3")
                st.pyplot(fig)

            with col1:
              st.write("### Correlation Heatmap")
              # Ensure "CustomerID" exists and remove the specific row
              if "CustomerID" in data.columns:
                  data = data[data["CustomerID"] != "7590-VHVEG"]
             # Select only numeric columns
              numeric_data = data.select_dtypes(include=["number"])
              # Plot the heatmap
              fig, ax = plt.subplots(figsize=(5, 3))
              sns.heatmap(numeric_data.corr(), annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
              st.pyplot(fig)


        elif dashboard_type == "Analytics":
            st.subheader("KPIs")
            attrition_rate = round((data["Churn"].value_counts(normalize=True).get("Yes", 0)) * 100, 2)
            avg_income = f"${data['MonthlyCharges'].mean():,.2f}"
            avg_clv = f"${data['TotalCharges'].mean():,.2f}"
            data_size = data.shape[0]

            col1, col2 = st.columns(2)
            with col1:
                st.metric(label="Attrition Rate", value=f"{attrition_rate}%")
                st.metric(label="Avg Monthly Income", value=avg_income)
            with col2:
                st.metric(label="Avg Customer Lifetime Value", value=avg_clv)
                st.metric(label="Data Size", value=data_size)

    except Exception as e:
        st.error(f"An error occurred: {e}")

# Predict Page
def preprocess_input(input_df, trained_features):
    """
    Preprocess the input data to match the trained model's feature set.
    - Adds missing features with default values in one step.
    - Ensures the column order matches the trained model.
    """
    # Create dummy variables for categorical data
    input_df = pd.get_dummies(input_df, drop_first=True)

    # Identify missing and extra features
    missing_features = [feature for feature in trained_features if feature not in input_df.columns]
    extra_features = [feature for feature in input_df.columns if feature not in trained_features]

    # Add missing features with default value 0
    if missing_features:
        missing_df = pd.DataFrame(0, index=input_df.index, columns=missing_features)
        input_df = pd.concat([input_df, missing_df], axis=1)

    # Drop extra features
    input_df = input_df.drop(columns=extra_features, errors="ignore")

    # Ensure the column order matches the trained model
    input_df = input_df[trained_features]

    # Return a de-fragmented DataFrame
    return input_df.copy()



def predict_page():
    st.title("Predict Customer Churn")

    if model is None:
        st.error("The prediction model is not loaded. Please check the model file.")
        return

    # Input customer details
    st.subheader("Enter Customer Details")
    gender = st.selectbox("Gender", ["Male", "Female"])
    senior_citizen = st.selectbox("Senior Citizen", ["Yes", "No"])
    partner = st.selectbox("Partner", ["Yes", "No"])
    dependents = st.selectbox("Dependents", ["Yes", "No"])
    tenure = st.slider("Tenure (Months)", 0, 72, 12)
    phone_service = st.selectbox("Phone Service", ["Yes", "No"])
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

    # Prepare input data for prediction
    input_data = {
        "Gender": gender,
        "SeniorCitizen": 1 if senior_citizen == "Yes" else 0,
        "Partner": 1 if partner == "Yes" else 0,
        "Dependents": 1 if dependents == "Yes" else 0,
        "Tenure": tenure,
        "PhoneService": 1 if phone_service == "Yes" else 0,
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

    input_df = pd.DataFrame([input_data])

    # Preprocess the input to align with the model's features
    trained_features = model.feature_names_in_  # Features used during model training
    input_df = preprocess_input(input_df, trained_features)

    # Predict
    if st.button("Predict"):
        prediction = model.predict(input_df)[0]
        prediction_label = "This customer is likely to churn" if prediction == 1 else "This customer is not likely to churn"
        st.success(f"Prediction: {prediction_label}")


# History Page
def history_page():
    st.title("Prediction History")
    try:
        if os.path.exists("history.csv"):
            history = pd.read_csv("history.csv")
            st.dataframe(history)
        else:
            st.warning("No prediction history found.")
    except Exception as e:
        st.error(f"An error occurred: {e}")

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
