import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import shap
import io

# Set page configuration
st.set_page_config(page_title="Telco Customer Churn Analysis", layout="wide", page_icon="📊")

# Application title and introduction
st.title("Telco Customer Churn Analysis Dashboard")
st.markdown("""
This application provides interactive analysis of the Telco Customer Churn dataset. 
Explore customer demographics, service usage patterns, and predict churn probability.
""")

# Function to load data
@st.cache_data
def load_data():
    try:
        # In a real app, you would use:
         df = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')
        # For this example, we'll use a text input showing the first few row
        
        # Data preprocessing
        # Convert TotalCharges to numeric
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
        # Fill missing values
        df['TotalCharges'].fillna(0, inplace=True)
        # Convert SeniorCitizen from 0/1 to No/Yes
        df['SeniorCitizen'] = df['SeniorCitizen'].map({0: 'No', 1: 'Yes'})
        
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

# Sidebar for data input and navigation
with st.sidebar:
    st.header("Navigation")
    page = st.radio("Select a Page", 
                    ["Data Overview", 
                     "Customer Demographics", 
                     "Service Usage Analysis", 
                     "Churn Prediction", 
                     "Customer Profile Analysis"])
    
    st.header("Data Input")
    if 'data_input' not in st.session_state:
        st.session_state.data_input = st.text_area(
            "Paste your data here (tab-separated format) or upload a file:",
            height=200
        )
    else:
        st.session_state.data_input = st.text_area(
            "Paste your data here (tab-separated format) or upload a file:",
            st.session_state.data_input,
            height=200
        )
    
    uploaded_file = st.file_uploader("Or upload CSV file", type=['csv'])
    if uploaded_file is not None:
        st.session_state.data_input = uploaded_file.getvalue().decode("utf-8")
        st.success("File uploaded successfully!")

# Load the data
df = load_data()

# Main content based on selected page
if df is not None:
    if page == "Data Overview":
        st.header("Data Overview")
        
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Dataset Information")
            st.write(f"Total Records: {df.shape[0]}")
            st.write(f"Total Features: {df.shape[1]}")
            st.write(f"Churn Rate: {df['Churn'].value_counts(normalize=True)['Yes']*100:.2f}%")
            
        with col2:
            st.subheader ("Sample Data")
            st.dataframe(df.head())

    elif page == "Customer Demographics":
        st.header("Customer Demographics")
        gender_count = df['gender'].value_counts()
        st.bar_chart(gender_count)

        senior_citizen_count = df['SeniorCitizen'].value_counts()
        st.bar_chart(senior_citizen_count)

        partner_count = df['Partner'].value_counts()
        st.bar_chart(partner_count)

    elif page == "Service Usage Analysis":
        st.header("Service Usage Analysis")
        internet_service_count = df['InternetService'].value_counts()
        st.bar_chart(internet_service_count)

        phone_service_count = df['PhoneService'].value_counts()
        st.bar_chart(phone_service_count)

        tenure_churn = df.groupby('tenure')['Churn'].value_counts().unstack().fillna(0)
        tenure_churn.plot(kind='bar', stacked=True)
        st.pyplot(plt)

    elif page == "Churn Prediction":
        st.header("Churn Prediction Model")
        X = df.drop(columns=['Churn'])
        y = df['Churn'].map({'Yes': 1, 'No': 0})

        # Preprocessing
        numeric_features = ['tenure', 'MonthlyCharges', 'TotalCharges']
        categorical_features = ['gender', 'SeniorCitizen', 'Partner', 'Dependents', 'PhoneService', 
                                'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup', 
                                'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies', 
                                'Contract', 'PaperlessBilling', 'PaymentMethod']

        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), numeric_features),
                ('cat', OneHotEncoder(), categorical_features)
            ])

        X_processed = preprocessor.fit_transform(X)
        X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=0.2, random_state=42)

        model = RandomForestClassifier()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        st.subheader("Model Performance")
        st.write(f"Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")
        st.text(classification_report(y_test, y_pred))

    elif page == "Customer Profile Analysis":
        st.header("Customer Profile Analysis")
        selected_customer = st.selectbox("Select Customer ID", df['customerID'].unique())
        customer_data = df[df['customerID'] == selected_customer]
        st.write(customer_data)

        st.subheader("Churn Risk Assessment")
        if not customer_data.empty:
            features = customer_data.drop(columns=['Churn'])
            features_processed = preprocessor.transform(features)
            churn_probability = model.predict_proba(features_processed)[:, 1]
            st.write(f"Churn Probability: {churn_probability[0] * 100:.2f}%") 
        else:
            st.write("No data available for the selected customer.")

        st.subheader("Service Usage Trends")
        usage_data = df.groupby('InternetService').agg({'MonthlyCharges': 'mean', 'Churn': 'mean'}).reset_index()
        fig = px.bar(usage_data, x='InternetService', y='MonthlyCharges', color='Churn', 
                     title="Average Monthly Charges by Internet Service Type")
        st.plotly_chart(fig)

        st.subheader("Interactive Data Filtering")
        filter_options = st.multiselect("Select Features to Filter", df.columns.tolist())
        if filter_options:
            filtered_data = df[filter_options]
            st.dataframe(filtered_data)

# Footer
st.markdown("""
---
This application is built for educational purposes to analyze customer churn in the telecommunications industry. 
Feel free to explore the data and insights!
""")
