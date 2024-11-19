import streamlit as st
import pandas as pd
 
st.write("""pip install -r requirements.txt
# My first app
Hello *world!*
""")
sepal_height = st.number_input("Ënter sepal height")
sepal_width = st.number_input("Ënter sepal width")

petal_height = st.number_input("Ënter petal height")
petal_width = st.number_input("Ënter petal width") 

if st.button("Predict"):
    st.text(f"The iris has been classified as:  '{''}'.")