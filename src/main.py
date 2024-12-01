import streamlit as st
import pandas as pd
import os
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px


# Load the trained model
@st.cache_resource

@st.cache_resource
def load_model():
    try:
        model_path = os.path.join(os.getcwd(), "models", "churn_model.pkl")
       # st.write(f"Attempting to load model from: {model_path}")  # Debugging output
        with open(model_path, 'rb') as file:
            model = pickle.load(file)
        return model
    except FileNotFoundError:
        st.error(f"Model file not found at {model_path}. Please upload the file.")
        return None
    except Exception as e:
        st.error(f"An unexpected error occurred while loading the model: {e}")
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

# Sidebar Navigation with Logout Button
def navigation():
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Home", "View Data", "Dashboard", "Predict", "History"])

    # Add a Logout Button
    if st.sidebar.button("Logout"):
        st.session_state["authenticated"] = False
        st.session_state["navigate_to_login"] = True
        st.rerun()

    return page

# Homepage
def homepage():
    st.title("Welcome to the Telco Churn Insight Application")
    st.subheader("Churn Predictor")
    st.markdown("""
    This app predicts whether an customer will stop patronizing a Telco company or not.
    """)

    col1, col2 = st.columns(2)

    with col1:
        st.write("### Key Features")
        st.markdown("""
        - **View Data**: View Telco Churn Database.
        - **Dashboard**: Contains data visualizations and KPIs.
        - **Predict**: Allows you to make predictions in real time.
        """)

        st.write("### User Benefits")
        st.markdown("""
        - Make data-driven decisions effortlessly.
        - Harness the power of machine learning without complexity.
        """)

    with col2:
        st.write("### How to Run Application")
        st.markdown("""
        1. Activate your virtual environment:
        ```
        activate virtual environment
        env/scripts/activate
        ```
        2. Run the app:
        ```
        streamlit run main.py
        ```
        """)

        st.write("### Need Help?")
        st.markdown("""
        - Contact us at **millicent.agyir@azubiafrica.org**.
        - Visit our [GitHub Repository](https://github.com/MillicentAgyir/Streamlit-ML-App_Classification).
        """)

# View Data Page
def view_data_page():
    st.title("Data from Telcos")
    
    # Training Dataset
    st.subheader("Training Dataset")
    training_data = load_and_clean_data("Datasets/cleaned_combined_dataset.csv")
    if training_data is not None:
        st.dataframe(training_data)



    # Test Dataset
    # Test Dataset
    st.subheader("Test Dataset")
    try:
      test_data_path = "Datasets/TestData.csv"  # Use relative paths
      with open(test_data_path, 'r') as file:  # Check if the file exists
        test_dataset = pd.read_csv(test_data_path, delimiter=";")  # Correct delimiter
        st.dataframe(test_dataset)
    except FileNotFoundError:
      st.error("Test dataset not found. Please upload or include the file.")
    except Exception as e:
     st.error(f"An error occurred while loading the test dataset: {e}")



    # Feature Description
    with st.expander("Expand to learn about dataset features"):
      st.write(
        """
        ### Feature Description:
        - **CustomerID**: Unique identifier for each customer.
        - **Gender**: Gender of the customer (Male/Female).
        - **SeniorCitizen**: Whether the customer is a senior citizen (1: Yes, 0: No).
        - **Partner**: Whether the customer has a partner (Yes/No).
        - **Dependents**: Whether the customer has dependents (Yes/No).
        - **Tenure**: Number of months the customer has stayed with the company.
        - **PhoneService**: Whether the customer has a phone service (Yes/No).
        - **MultipleLines**: Whether the customer has multiple lines (Yes/No).
        - **InternetService**: Type of internet service (DSL/Fiber optic/No).
        - **OnlineSecurity**: Whether the customer has online security service (Yes/No).
        - **OnlineBackup**: Whether the customer has online backup service (Yes/No).
        - **DeviceProtection**: Whether the customer has device protection service (Yes/No).
        - **TechSupport**: Whether the customer has tech support service (Yes/No).
        - **StreamingTV**: Whether the customer has streaming TV service (Yes/No).
        - **StreamingMovies**: Whether the customer has streaming movies service (Yes/No).
        - **Contract**: The contract term (Month-to-month/One year/Two year).
        - **PaperlessBilling**: Whether the customer has paperless billing (Yes/No).
        - **PaymentMethod**: The payment method (Electronic check/Mailed check/Bank transfer/Credit card).
        - **MonthlyCharges**: Monthly charges paid by the customer.
        - **TotalCharges**: Total charges incurred by the customer.
        """
    )


# Dashboard Page
def dashboard_page():
    st.title("Dashboard")
    dashboard_type = st.selectbox("Select Dashboard Type", ["EDA", "Analytics"])

    try:
        # Dynamically construct the dataset path
        data_path = os.path.join(os.getcwd(), "Datasets", "cleaned_combined_dataset.csv")

        # Check if the dataset exists
        if not os.path.exists(data_path):
            st.error(f"Dataset not found at {data_path}. Please upload or include the file.")
            return

        # Load and clean the dataset
        data = pd.read_csv(data_path)
        if "CustomerID" in data.columns:
            data = data[data["CustomerID"] != "7590-VHVEG"]  # Remove specific row

        if "Churn" in data.columns:
            data["Churn"] = data["Churn"].replace({True: "Yes", False: "No"})  # Replace True/False with Yes/No

        if data.empty:
            st.error("The dataset is empty after cleaning.")
            return

        if dashboard_type == "EDA":
            st.subheader("Exploratory Data Analysis")
            col1, col2 = st.columns(2)

            # Gender Distribution
            if "gender" in data.columns:
                with col1:
                    st.write("### Gender Distribution")
                    gender_counts = data["gender"].value_counts()
                    gender_pie = px.pie(
                        values=gender_counts.values,
                        names=gender_counts.index,
                        color_discrete_sequence=px.colors.sequential.RdBu,
                    )
                    st.plotly_chart(gender_pie, use_container_width=True)

            # Outliers in Numerical Data
            if not data.select_dtypes(include=["number"]).empty:
                with col2:
                    st.write("### Outliers in Numerical Data")
                    fig, ax = plt.subplots(figsize=(10, 8))
                    sns.boxplot(data=data.select_dtypes(include=["number"]), ax=ax, palette="Set3")
                    st.pyplot(fig)

            # Correlation Heatmap
            if not data.select_dtypes(include=["number"]).empty:
                with col1:
                    st.write("### Correlation Heatmap")
                    numeric_data = data.select_dtypes(include=["number"])
                    fig, ax = plt.subplots(figsize=(5, 3))
                    sns.heatmap(numeric_data.corr(), annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
                    st.pyplot(fig)

        elif dashboard_type == "Analytics":
            st.subheader("KPIs")

            # Ensure required columns exist
            if {"Churn", "MonthlyCharges", "TotalCharges"}.issubset(data.columns):
                churn_rate = round((data["Churn"].value_counts(normalize=True).get("Yes", 0)) * 100, 2)
                avg_income = f"${data['MonthlyCharges'].mean():,.2f}"
                avg_clv = f"${data['TotalCharges'].mean():,.2f}"
                data_size = data.shape[0]

                col1, col2 = st.columns(2)
                with col1:
                    st.metric(label="Churn Rate", value=f"{churn_rate}%")
                    st.metric(label="Avg Monthly Income", value=avg_income)
                with col2:
                    st.metric(label="Avg Customer Lifetime Value", value=avg_clv)
                    st.metric(label="Data Size", value=data_size)
            else:
                st.warning("Required columns for analytics (Churn, MonthlyCharges, TotalCharges) are missing.")
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
    input_data = {
        "Gender": st.selectbox("Gender", ["Male", "Female"]),
        "SeniorCitizen": 1 if st.selectbox("Senior Citizen", ["Yes", "No"]) == "Yes" else 0,
        "Partner": 1 if st.selectbox("Partner", ["Yes", "No"]) == "Yes" else 0,
        "Dependents": 1 if st.selectbox("Dependents", ["Yes", "No"]) == "Yes" else 0,
        "Tenure": st.slider("Tenure (Months)",  value=12, step=1),
        #"tenure = st.number_input("Tenure (Months)", value=12, step=1),
        "PhoneService": 1 if st.selectbox("Phone Service", ["Yes", "No"]) == "Yes" else 0,
        "InternetService": st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"]),
        "OnlineSecurity": st.selectbox("Online Security", ["Yes", "No", "No internet service"]),
        "OnlineBackup": st.selectbox("Online Backup", ["Yes", "No", "No internet service"]),
        "DeviceProtection": st.selectbox("Device Protection", ["Yes", "No", "No internet service"]),
        "TechSupport": st.selectbox("Tech Support", ["Yes", "No", "No internet service"]),
        "StreamingTV": st.selectbox("Streaming TV", ["Yes", "No", "No internet service"]),
        "StreamingMovies": st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"]),
        "Contract": st.selectbox("Contract", ["Month-to-month", "One year", "Two year"]),
        "PaperlessBilling": 1 if st.selectbox("Paperless Billing", ["Yes", "No"]) == "Yes" else 0,
        "PaymentMethod": st.selectbox(
            "Payment Method",
            ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"]
        ),
        "MonthlyCharges": st.number_input("Monthly Charges", value=0.00, step=0.1),
        "TotalCharges": st.number_input("Total Charges", value=0.00, step=0.1),
    }

    input_df = pd.DataFrame([input_data])

    # Preprocess the input
    trained_features = model.feature_names_in_
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

if "navigate_to_login" in st.session_state and st.session_state["navigate_to_login"]:
    authenticate()
else:
    main_app()
