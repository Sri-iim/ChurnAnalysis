import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report

# Load the dataset
@st.cache_data
def load_data():
    url = "WA_Fn-UseC_-Telco-Customer-Churn.csv"  # Replace with your actual path or URL
    df = pd.read_csv(url)
    df.columns = df.columns.str.strip()
    return df

df = load_data()

# Data Preprocessing
def preprocess_data(df):
    if 'customerID' in df.columns:
        df.drop(columns=['customerID'], inplace=True)

    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['tenure'] = df['tenure'].astype(int)
    categorical_columns = ['gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines',
                           'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
                           'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract',
                           'PaperlessBilling', 'PaymentMethod', 'Churn']
    df[categorical_columns] = df[categorical_columns].astype('category')
    total_charges_median = df['TotalCharges'].median()
    df['TotalCharges'] = df['TotalCharges'].fillna(total_charges_median)
    for col in categorical_columns:
        if df[col].isnull().any():
            mode_value = df[col].mode()[0]
            df[col] = df[col].fillna(mode_value)
    df['PhoneService_MultipleLines'] = df.apply(
        lambda row: 'No Phone Service' if row['PhoneService'] == 'No' else row['MultipleLines'], axis=1
    )
    services = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']
    df['TotalServices'] = df[services].apply(lambda row: (row == 'Yes').sum(), axis=1)

    def tenure_category(tenure):
        if tenure <= 12:
            return '0-12 months'
        elif tenure <= 24:
            return '12-24 months'
        elif tenure <= 36:
            return '24-36 months'
        elif tenure <= 48:
            return '36-48 months'
        elif tenure <= 60:
            return '48-60 months'
        else:
            return '60+ months'

    df['TenureCategory'] = df['tenure'].apply(tenure_category)
    df['AverageMonthlySpend'] = df.apply(
        lambda row: row['TotalCharges'] / row['tenure'] if row['tenure'] > 0 else 0, axis=1
    )
    df.drop(columns=['PhoneService', 'MultipleLines'] + services, inplace=True)
    yes_no_columns = ['Partner', 'Dependents', 'PaperlessBilling', 'Churn']
    for col in yes_no_columns:
        df[col] = df[col].map({'Yes': 1, 'No': 0})
        mode_value = df[col].mode()[0]
        df[col] = df[col].fillna(mode_value).astype(int)
    numerical_columns = ['MonthlyCharges', 'TotalCharges', 'AverageMonthlySpend']
    for col in numerical_columns:
        median_value = df[col].median()
        df[col] = df[col].fillna(median_value)
    df['PhoneService_MultipleLines'] = df['PhoneService_MultipleLines'].apply(
        lambda x: 1 if x != 'No Phone Service' else 0
    ).astype(int)
    return df

df = preprocess_data(df.copy())

# Additional Feature - Customer Profile Summary
st.subheader("Customer Profile Summary")
st.write(df.describe(include='all'))

# Additional Feature - Service Usage Trend Analysis
st.subheader("Service Usage Trend Analysis")
fig5, ax5 = plt.subplots()
sns.histplot(df['TotalServices'], kde=True, bins=10, ax=ax5)
st.pyplot(fig5)

# Additional Feature - Churn Risk Assessment Indicator
def churn_risk(row):
    if row['TotalServices'] < 2 and row['tenure'] < 12 and row['MonthlyCharges'] > 70:
        return 'High Risk'
    elif row['TotalServices'] >= 2 and row['tenure'] > 24:
        return 'Low Risk'
    else:
        return 'Medium Risk'

df['ChurnRisk'] = df.apply(churn_risk, axis=1)
st.subheader("Churn Risk Assessment")
st.write(df[['ChurnRisk', 'Churn']].value_counts())
