import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report

# Streamlit Page Config
st.set_page_config(page_title="Customer Churn Prediction", layout="wide")

# Load the dataset
@st.cache_data
def load_data():
    url = "WA_Fn-UseC_-Telco-Customer-Churn.csv"  # Replace with your actual path or URL
    df = pd.read_csv(url)
    df.columns = df.columns.str.strip()
    return df

df = load_data()

# Sidebar - Data Filters
st.sidebar.header("Filter Options")
gender_filter = st.sidebar.selectbox("Gender", ['All'] + df['gender'].unique().tolist())
partner_filter = st.sidebar.selectbox("Partner Status", ['All'] + df['Partner'].unique().tolist())
contract_filter = st.sidebar.selectbox("Contract Type", ['All'] + df['Contract'].unique().tolist())

if gender_filter != 'All':
    df = df[df['gender'] == gender_filter]
if partner_filter != 'All':
    df = df[df['Partner'] == partner_filter]
if contract_filter != 'All':
    df = df[df['Contract'] == contract_filter]

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
    df['TotalCharges'] = df['TotalCharges'].fillna(df['TotalCharges'].median())
    
    # Feature Engineering
    df['PhoneService_MultipleLines'] = df.apply(lambda row: 'No Phone Service' if row['PhoneService'] == 'No' else row['MultipleLines'], axis=1)
    df['TotalServices'] = df[['OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']].apply(lambda row: (row == 'Yes').sum(), axis=1)
    df['AverageMonthlySpend'] = df.apply(lambda row: row['TotalCharges'] / row['tenure'] if row['tenure'] > 0 else 0, axis=1)
    
    # Binary Mapping
    yes_no_columns = ['Partner', 'Dependents', 'PaperlessBilling', 'Churn']
    for col in yes_no_columns:
        df[col] = df[col].map({'Yes': 1, 'No': 0}).astype(int)

    df['PhoneService_MultipleLines'] = df['PhoneService_MultipleLines'].apply(lambda x: 1 if x != 'No Phone Service' else 0).astype(int)
    
    return df

df = preprocess_data(df.copy())

# Title & Overview
st.title("📊 Customer Churn Prediction Dashboard")
st.markdown("This dashboard provides insights into customer churn behavior and allows you to predict churn using machine learning models.")

# Data Exploration
st.subheader("📋 Data Overview")
if st.checkbox("Show Raw Data"):
    st.dataframe(df)

# Visualizations
st.subheader("📈 Data Visualizations")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Churn Distribution")
    fig, ax = plt.subplots()
    sns.countplot(x='Churn', data=df, palette=['#fc5a50', '#50c878'], ax=ax)
    for p in ax.patches:
        ax.annotate(f'{int(p.get_height())}', (p.get_x() + p.get_width() / 2., p.get_height()), ha='center', fontsize=10, color='black', xytext=(0, 5), textcoords='offset points')
    st.pyplot(fig)

with col2:
    st.subheader("Churn by Tenure Category")
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.countplot(x='tenure', hue='Churn', data=df, palette=['#fc5a50', '#50c878'], ax=ax)
    st.pyplot(fig)

st.subheader("🔍 Correlation Heatmap")
fig, ax = plt.subplots(figsize=(10, 6))
df_encoded = pd.get_dummies(df, drop_first=True)
sns.heatmap(df_encoded.corr(), annot=True, cmap='coolwarm', fmt='.2f', ax=ax)
st.pyplot(fig)

# Machine Learning Model Training
st.subheader("🤖 Churn Prediction Model")

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



# Improved Model Evaluation with Visuals
def evaluate_model(model, X_test, y_test, model_name):
    y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred)
    
    # Display Results in a More Visual Manner
    st.subheader(f"🔍 {model_name} Performance")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("📌 Accuracy", f"{accuracy:.4f}")
        st.progress(float(accuracy))
        
    with col2:
        st.metric("🎯 Precision", f"{precision:.4f}")
        st.progress(float(precision))
        
    with col3:
        st.metric("📢 Recall", f"{recall:.4f}")
        st.progress(float(recall))
        
    with col4:
        st.metric("⚡ F1 Score", f"{f1:.4f}")
        st.progress(float(f1))
    
    # Confusion Matrix Heatmap
    st.subheader(f"📊 Confusion Matrix - {model_name}")
    fig, ax = plt.subplots(figsize=(5, 4))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["No Churn", "Churn"], yticklabels=["No Churn", "Churn"], ax=ax)
    st.pyplot(fig)
    
    # Classification Report Summary
    st.text("📄 Classification Report")
    st.text(classification_report(y_test, y_pred))

# Display evaluation for both models
st.subheader("📊 Model Performance Summary")
evaluate_model(log_reg, X_test, y_test, "Logistic Regression")
evaluate_model(rf, X_test, y_test, "Random Forest")

# # Model Evaluation
# def evaluate_model(model, X_test, y_test, model_name):
#     y_pred = model.predict(X_test)
#     accuracy = accuracy_score(y_test, y_pred)
#     precision = precision_score(y_test, y_pred)
#     recall = recall_score(y_test, y_pred)
#     f1 = f1_score(y_test, y_pred)
#     roc_auc = roc_auc_score(y_test, y_pred)

#     st.write(f"### {model_name} Performance")
#     st.metric("Accuracy", f"{accuracy:.4f}")
#     st.metric("Precision", f"{precision:.4f}")
#     st.metric("Recall", f"{recall:.4f}")
#     st.metric("F1 Score", f"{f1:.4f}")
#     st.metric("ROC AUC", f"{roc_auc:.4f}")

# st.subheader("📊 Model Performance")
# evaluate_model(log_reg, X_test, y_test, "Logistic Regression")
# evaluate_model(rf, X_test, y_test, "Random Forest")

# Prediction on Sample Data
st.subheader("🎯 Churn Prediction")
if st.button("Predict Churn"):
    sample_data = X_test.iloc[:1]
    pred_lr = log_reg.predict(sample_data)[0]
    pred_rf = rf.predict(sample_data)[0]
    
    st.write(f"**Logistic Regression Prediction:** {'Churn' if pred_lr == 1 else 'No Churn'}")
    st.write(f"**Random Forest Prediction:** {'Churn' if pred_rf == 1 else 'No Churn'}")

# Customer Profile Summary
st.subheader("🧩 Customer Profile Summary")
st.dataframe(df.describe(include='all'))

st.subheader("📊 Service Usage Trend")
fig, ax = plt.subplots()
sns.histplot(df['TotalServices'], kde=True, bins=10, color="#3b5998", ax=ax)
st.pyplot(fig)

