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

# Streamlit App
st.title("Telco Customer Churn Prediction")

# Data Exploration
st.subheader("Data Exploration")
if st.checkbox("Show Raw Data"):
    st.write(df)

# Interactive Filters
st.sidebar.header("Filter Options")
gender_filter = st.sidebar.selectbox("Select Gender", options=['All'] + df['gender'].unique().tolist())
if gender_filter != 'All':
    df = df[df['gender'] == gender_filter]

partner_filter = st.sidebar.selectbox("Select Partner Status", options=['All'] + df['Partner'].unique().tolist())
if partner_filter != 'All':
    df = df[df['Partner'] == partner_filter]

# Visualizations
st.subheader("Visualizations")

# Churn Distribution
st.subheader("Churn Distribution")
fig1, ax1 = plt.subplots()
sns.countplot(x='Churn', data=df, ax=ax1)
for p in ax1.patches:
    ax1.annotate(f'{int(p.get_height())}', (p.get_x() + p.get_width() / 2., p.get_height()),
                 ha='center', va='center', fontsize=10, color='black', xytext=(0, 5),
                 textcoords='offset points')
st.pyplot(fig1)

# Churn by Tenure Category
st.subheader("Churn by Tenure Category")
fig2, ax2 = plt.subplots(figsize=(10, 6))
sns.countplot(x='TenureCategory', hue='Churn', data=df, ax=ax2)
for p in ax2.patches:
    ax2.annotate(f'{int(p.get_height())}', (p.get_x() + p.get_width() / 2., p.get_height()),
                 ha='center', va='center', fontsize=10, color='black', xytext=(0, 5),
                 textcoords='offset points')
st.pyplot(fig2)

# Churn by Total Services
st.subheader("Churn by Total Services")
fig3, ax3 = plt.subplots()
sns.boxplot(x='Churn', y='TotalServices', data=df, ax=ax3)
st.pyplot(fig3)

# Correlation Heatmap
st.subheader("Correlation Heatmap")
fig4, ax4 = plt.subplots(figsize=(12, 8))
df_encoded = pd.get_dummies(df, drop_first=True)
correlation_matrix = df_encoded.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', ax=ax4)
st.pyplot(fig4)

# Real-Time Churn Prediction Model
st.subheader("Churn Prediction")

# User Input for Prediction
st.sidebar.header("Customer Input")
def user_input_features():
    gender = st.sidebar.selectbox("Gender", options=['Male', 'Female'])
    partner = st.sidebar.selectbox("Partner", options=['Yes', 'No'])
    dependents = st.sidebar.selectbox("Dependents", options=['Yes', 'No'])
    tenure = st.sidebar.slider("Tenure (months)", 0, 72, 12)
    monthly_charges = st.sidebar.number_input("Monthly Charges", min_value=0.0, max_value=100.0, value=30.0)
    total_charges = st.sidebar.number_input("Total Charges", min_value=0.0, value=100.0)

    # Default input DataFrame
    input_data = pd.DataFrame({
        'gender': [gender],
        'Partner': [partner],
        'Dependents': [dependents],
        'tenure': [tenure],
        'MonthlyCharges': [monthly_charges],
        'TotalCharges': [total_charges]
    })

    # Add missing categorical columns with default values
    expected_columns = ['PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity',
                        'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV',
                        'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod', 'Churn']

    for col in expected_columns:
        if col not in input_data.columns:
            input_data[col] = 'No'  # Default assumption

    # Additional features
    input_data['PhoneService_MultipleLines'] = 1  # Assuming user has phone service
    input_data['TotalServices'] = 0  # Placeholder for total services
    input_data['AverageMonthlySpend'] = input_data['TotalCharges'] / input_data['tenure'].replace(0, 1)

    return input_data

input_data = user_input_features()
input_data['PhoneService_MultipleLines'] = 1  # Assuming user has phone service
input_data['TotalServices'] = 0  # Placeholder for total services
input_data['AverageMonthlySpend'] = input_data['TotalCharges'] / input_data['tenure'].replace(0, 1)

# Preprocess input data
input_data = preprocess_data(input_data)


log_reg = LogisticRegression(random_state=42)
rf = RandomForestClassifier(random_state=42)

# log_reg.fit(X_train, y_train)
# rf.fit(X_train, y_train)
# Model Prediction
if st.button("Predict Churn"):
    prediction_log_reg = log_reg.predict(input_data)
    prediction_rf = rf.predict(input_data)
    st.write(f"Logistic Regression Prediction: {'Churn' if prediction_log_reg[0] == 1 else 'No Churn'}")
    st.write(f"Random Forest Prediction: {'Churn' if prediction_rf[0] == 1 else 'No Churn'}")

# Model Training and Evaluation
st.subheader("Model Training and Evaluation")

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

def evaluate_model(model, X_train, X_test, y_train, y_test, model_name):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred)
    st.write(f"Model: {model_name}")
    st.write(f"Accuracy: {accuracy:.4f}")
    st.write(f"Precision: {precision:.4f}")
    st.write(f"Recall: {recall:.4f}")
    st.write(f"F1-Score: {f1:.4f}")
    st.write(f"ROC-AUC: {roc_auc:.4f}")
    st.write("Confusion Matrix:")
    st.write(confusion_matrix(y_test, y_pred))
    st.write("Classification Report:")
    st.write(classification_report(y_test, y_pred))

evaluate_model(log_reg, X_train, X_test, y_train, y_test, "Logistic Regression")
evaluate_model(rf, X_train, X_test, y_train, y_test, "Random Forest")

# Churn Prediction
st.subheader("Churn Prediction")
if st.button("Predict Churn"):
    prediction_log_reg = log_reg.predict(X_test.iloc[:1])
    prediction_rf = rf.predict(X_test.iloc[:1])
    st.write(f"Logistic Regression Prediction: {'Churn' if prediction_log_reg[0] == 1 else 'No Churn'}")
    st.write(f"Random Forest Prediction: {'Churn' if prediction_rf[0] == 1 else 'No Churn'}")


# log_reg = LogisticRegression(random_state=42)
# rf = RandomForestClassifier(random_state=42)

# st.subheader("Logistic Regression Evaluation")
# log_reg_model = evaluate_model(log_reg, X_train, X_test, y_train, y_test, "Logistic Regression")

# st.subheader("Random Forest Evaluation")
# rf_model = evaluate_model(rf, X_train, X_test, y_train, y_test, "Random Forest")

# Customer Profile Analysis
st.subheader("Customer Profile Analysis")
st.write("Analyze customer profiles based on selected filters and visualizations.")
# Add more interactive visualizations or insights based on customer profiles here

# Service Usage Trends
st.subheader("Service Usage Trends")
# Implement visualizations to show trends in service usage over time or by customer demographics

# Churn Risk Assessment
st.subheader("Churn Risk Assessment")
# Provide insights or visualizations that help assess the risk of churn based on various factors

# Interactive Data Filtering and Exploration
st.subheader("Interactive Data Filtering")
# Allow users to filter data based on various criteria and visualize the results dynamically

st.sidebar.header("Additional Filters")
# Add more filter options for users to explore the dataset further

# Finalize the Streamlit app
st.write("Explore the data, visualize trends, and predict churn risk effectively!")
