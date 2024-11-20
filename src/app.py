import streamlit as st
import pandas as pd
import pickle

# Load the trained model
with open('model/churn_model.pkl', 'rb') as file:
    model = pickle.load(file)

# Define the prediction function
def predict(features):
    # Convert features to a DataFrame
    input_df = pd.DataFrame([features])
    prediction = model.predict(input_df)
    return prediction[0]

# Streamlit app layout
st.title("Customer Churn Prediction App")
st.write("Enter the customer details below to predict churn.")

# Input features
age = st.number_input("Age", min_value=18, max_value=100, value=30)
monthly_income = st.number_input("Monthly Income", min_value=0, max_value=100000, value=5000)
tenure = st.number_input("Tenure (Months)", min_value=0, max_value=72, value=12)

# Create a dictionary of inputs
input_features = {
    'Age': age,
    'MonthlyIncome': monthly_income,
    'Tenure': tenure
}

# Display inputs
st.write("### Input Features")
st.write(input_features)

# Make predictions
if st.button("Predict Churn"):
    result = predict(input_features)
    if result == 1:
        st.error("This customer is likely to churn.")
    else:
        st.success("This customer is not likely to churn.")
