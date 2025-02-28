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
    url = "WA_Fn-UseC_-Telco-Customer-Churn.csv"
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
    df.fillna({'TotalCharges': df['TotalCharges'].median()}, inplace=True)
    
    categorical_columns = ['gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines',
                           'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
                           'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract',
                           'PaperlessBilling', 'PaymentMethod', 'Churn']
    df[categorical_columns] = df[categorical_columns].astype('category')
    
    services = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']
    df['TotalServices'] = df[services].apply(lambda row: (row == 'Yes').sum(), axis=1)
    df.drop(columns=services, inplace=True)
    
    df['AverageMonthlySpend'] = df.apply(lambda row: row['TotalCharges'] / row['tenure'] if row['tenure'] > 0 else 0, axis=1)
    
    yes_no_columns = ['Partner', 'Dependents', 'PaperlessBilling', 'Churn']
    for col in yes_no_columns:
        df[col] = df[col].map({'Yes': 1, 'No': 0})
    
    return df

df = preprocess_data(df.copy())

# Streamlit App
st.title("Telco Customer Churn Analysis")

# Sidebar Filters
st.sidebar.header("Filter Options")
gender_filter = st.sidebar.selectbox("Select Gender", ['All'] + df['gender'].unique().tolist())
partner_filter = st.sidebar.selectbox("Select Partner Status", ['All'] + df['Partner'].unique().tolist())
if gender_filter != 'All':
    df = df[df['gender'] == gender_filter]
if partner_filter != 'All':
    df = df[df['Partner'] == partner_filter]

# Data Exploration
st.subheader("Customer Profile Analysis")
st.write("### Monthly Spend by Customer Type")
fig1, ax1 = plt.subplots()
sns.boxplot(x='Churn', y='AverageMonthlySpend', data=df, ax=ax1)
st.pyplot(fig1)

st.write("### Service Usage Trends")
st.write("Percentage of Customers Using Each Service")
services_usage = df[['TotalServices', 'Churn']].groupby('TotalServices').size().reset_index(name='Count')
fig2, ax2 = plt.subplots()
sns.barplot(x='TotalServices', y='Count', data=services_usage, ax=ax2)
st.pyplot(fig2)

# Model Training
st.subheader("Churn Prediction Model")
X = df.drop(columns=['Churn'])
y = df['Churn']
X_encoded = pd.get_dummies(X, drop_first=True)
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
numerical_columns = ['tenure', 'MonthlyCharges', 'TotalCharges', 'AverageMonthlySpend', 'TotalServices']
X_train[numerical_columns] = scaler.fit_transform(X_train[numerical_columns])
X_test[numerical_columns] = scaler.transform(X_test[numerical_columns])

log_reg = LogisticRegression(random_state=42)
rf = RandomForestClassifier(random_state=42)
log_reg.fit(X_train, y_train)
rf.fit(X_train, y_train)

def evaluate_model(model, model_name):
    y_pred = model.predict(X_test)
    st.write(f"## {model_name} Performance")
    st.write(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    st.write(f"Precision: {precision_score(y_test, y_pred):.4f}")
    st.write(f"Recall: {recall_score(y_test, y_pred):.4f}")
    st.write(f"F1-Score: {f1_score(y_test, y_pred):.4f}")
    st.write("Confusion Matrix:")
    st.write(confusion_matrix(y_test, y_pred))
    st.write("Classification Report:")
    st.write(classification_report(y_test, y_pred))

evaluate_model(log_reg, "Logistic Regression")
evaluate_model(rf, "Random Forest")

# Churn Prediction
st.subheader("Churn Prediction for New Customers")
tenure = st.sidebar.slider("Tenure (months)", 0, 72, 12)
monthly_charges = st.sidebar.number_input("Monthly Charges", min_value=0.0, max_value=100.0, value=30.0)
total_charges = st.sidebar.number_input("Total Charges", min_value=0.0, value=100.0)
total_services = st.sidebar.slider("Total Services Used", 0, 6, 2)

def predict_churn():
    input_data = pd.DataFrame({
        'tenure': [tenure],
        'MonthlyCharges': [monthly_charges],
        'TotalCharges': [total_charges],
        'TotalServices': [total_services],
        'AverageMonthlySpend': [total_charges / tenure if tenure > 0 else 0]
    })
    input_data[numerical_columns] = scaler.transform(input_data[numerical_columns])
    pred_log_reg = log_reg.predict(input_data)
    pred_rf = rf.predict(input_data)
    st.write(f"Logistic Regression Prediction: {'Churn' if pred_log_reg[0] == 1 else 'No Churn'}")
    st.write(f"Random Forest Prediction: {'Churn' if pred_rf[0] == 1 else 'No Churn'}")

if st.button("Predict Churn"):
    predict_churn()
