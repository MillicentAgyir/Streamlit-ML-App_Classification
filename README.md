**Telco Customer Churn Prediction App**

📄 **Overview**

The Telco Customer Churn Prediction App is a Streamlit-based application designed to predict customer churn for telecommunications companies. The app uses a machine learning model to analyze customer data and provides insights such as KPIs, visualizations, and real-time predictions. It includes the following features:

**View Data:** Access the Telco dataset used for training and testing.

**Dashboard:** Visualize key metrics through interactive charts and plots.

**Predict:** Input customer data to predict churn likelihood.

**History:** Review past predictions for tracking and analysis.


📂 **Project Structure**

├── src/

│   ├── app_trial.py          # Main application file

├── models/

│   ├── churn_model.pkl # Pre-trained machine learning model

├── Datasets/

│   ├── cleaned_combined_dataset.csv # Training dataset

│   ├── TestData.csv                 # Test dataset

├── requirements.txt    # Required Python packages

├── README.md           # Project documentation



**⚙️ Features**

**Dashboard:** Explore data visualizations and customer insights.

**Predict Churn:** Make real-time predictions on customer churn.

**KPI Analysis:** Analyze key performance indicators like churn rate and average monthly income.


**🚀 How to Set Up the App Locally**

Follow the steps below to set up the Telco Churn Prediction App on your local machine:

**Prerequisites**

**Python:** Ensure Python 3.10 or higher is installed.

**Git:** Clone the repository or download the source code.

**Environment:** Set up a Python virtual environment.


**Steps to Set Up**

**Clone the repository:**

**git clone:** https://github.com/MillicentAgyir/Streamlit-ML-App_Classification

**Set up a virtual environment:**

python -m venv env

source env/bin/activate   # For Linux/Mac

.\env\Scripts\activate    # For Windows


**Install required dependencies:**

pip install -r requirements.txt

Place the datasets and model file:


Place cleaned_combined_dataset.csv and TestData.csv in the Datasets/ folder.

Place the churn_model.pkl file in the models/ folder.

**Run the Streamlit app:**
streamlit run src/app_trial.py

**Open the app in your browser:**
By default, the app will be available at http://localhost:8501.


**Login Credentials:**

**username :**  admin

**password:**  password123


**🛠️ Troubleshooting**


**Missing Model File:**

Ensure churn_model.pkl is placed in the models/ folder.

**Missing Dataset File:**

Place cleaned_combined_dataset.csv and TestData.csv in the Datasets/ folder.

**Dependency Issues:**

Run pip install -r requirements.txt to install all required dependencies.

**🧩 Key Features Breakdown**


**View Data**

Visualize the training and test datasets.

Expandable section to explore dataset column descriptions.


**Dashboard**

Interactive charts for:

Gender distribution

Correlation heatmap

Outliers detection

KPI insights such as churn rate and average income.


**Predict**

Input customer details to predict churn probability.

Real-time churn analysis using a pre-trained model.


**History**

View past prediction data for analysis.



For any questions or issues, feel free to contact me at millicent.agyir@azubiafrica.org.
