
import streamlit as st
import pandas as pd
import pickle
import datetime
import json

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
    st.write("*Please input customer details below:*")

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
    payment_method = st.selectbox("Payment Method", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])
    monthly_charges = st.number_input("Monthly Charges", 0.0, 200.0, 50.0)
    total_charges = st.number_input("Total Charges", 0.0, 10000.0, 500.0)

    # Input Features
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
    input_df = pd.DataFrame([input_features])

    # Preprocess Input Data
    @st.cache_data
    def preprocess_input(input_df, trained_features):
        # Apply one-hot encoding
        input_df = pd.get_dummies(input_df, drop_first=True)

        # Add missing columns
        for feature in trained_features:
            if feature not in input_df.columns:
                input_df[feature] = 0

        # Ensure column order matches the model's training data
        input_df = input_df[trained_features]

        return input_df

    # Get the feature names from the trained model
    trained_features = model.feature_names_in_

    # Preprocess input data
    input_df = preprocess_input(input_df, trained_features)

    # Predict Button
    if st.button("Predict"):
        # Get the prediction
        prediction = model.predict(input_df)[0]

        # Ensure all values in input_features are JSON serializable
        serializable_features = {k: int(v) if isinstance(v, bool) else v for k, v in input_features.items()}

        # Save prediction to history.csv
        with open("history.csv", "a") as f:
            row = {
                "Timestamp": str(datetime.datetime.now()),
                "Input Features": serializable_features,
                "Prediction": prediction
            }
            # Write each entry as a JSON object on a new line
            f.write(json.dumps(row) + "\n")

        # Display Prediction Result
        if prediction == 1:
            st.error("This customer is likely to churn.")
        else:
            st.success("This customer is not likely to churn.")

# History Page
elif page == "History":
    st.title("History Page")
    st.subheader("📋 Prediction History")

    try:
        # Read each line in history.csv as a JSON object
        with open("history.csv", "r") as f:
            history = [json.loads(line.strip()) for line in f]

        # Convert history to a DataFrame
        history_df = pd.DataFrame(history)

        # Expand Input Features into separate columns
        input_features_df = pd.json_normalize(history_df["Input Features"])
        history_df = pd.concat([history_df.drop(columns=["Input Features"]), input_features_df], axis=1)

        # Convert Prediction to readable format
        history_df["Prediction"] = history_df["Prediction"].apply(lambda x: "Yes" if x == 1 else "No")

        # Display the History Table
        st.dataframe(history_df)

    except FileNotFoundError:
        st.error("No history found. Make some predictions first!")
    except json.JSONDecodeError as e:
        st.error(f"An error occurred while parsing the history file: {e}")
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")

              

