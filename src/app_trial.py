import streamlit as st
import pandas as pd
import os
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
    #st.subheader(" ")

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
                if "CustomerID" in data.columns:
                    data = data[data["CustomerID"] != "7590-VHVEG"]
                numeric_data = data.select_dtypes(include=["number"])
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
def predict_page():
    st.title("Predict Customer Churn")

    if model is None:
        st.error("The prediction model is not loaded. Please check the model file.")
        return

    st.subheader("Enter Customer Details")
    gender = st.selectbox("Gender", ["Male", "Female"])
    senior_citizen = st.selectbox("Senior Citizen", ["Yes", "No"])
    partner = st.selectbox("Partner", ["Yes", "No"])
    dependents = st.selectbox("Dependents", ["Yes", "No"])
    tenure = st.slider("Tenure (Months)", 0, 72, 12)
    phone_service = st.selectbox("Phone Service", ["Yes", "No"])
    internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    monthly_charges = st.number_input("Monthly Charges", 0.0, 200.0, 50.0)
    total_charges = st.number_input("Total Charges", 0.0, 10000.0, 500.0)

    input_data = {
        "Gender": gender,
        "SeniorCitizen": 1 if senior_citizen == "Yes" else 0,
        "Partner": 1 if partner == "Yes" else 0,
        "Dependents": 1 if dependents == "Yes" else 0,
        "Tenure": tenure,
        "PhoneService": 1 if phone_service == "Yes" else 0,
        "InternetService": internet_service,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges,
    }

    input_df = pd.DataFrame([input_data])

    if st.button("Predict"):
        prediction = model.predict(input_df)[0]
        st.success(f"Prediction: {'This customer is likely to churn' if prediction == 1 else 'This customer is not likely to churn'}")

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
