import streamlit as st
import pandas as pd
 
st.write("""# Telco Customer Churn Prediction
Hello *world!*
""")
sepal_height = st.number_input("Enter sepal height")
sepal_width = st.number_input("Enter sepal width")

petal_height = st.number_input("Enter petal height")
petal_width = st.number_input("Enter petal width") 

if st.button("Predict"):
    st.text(f"The iris has been classified as:  '{''}'.")