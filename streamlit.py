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
    return df

df = load_data()

# Data Preprocessing and Feature Engineering
# Print columns for debugging
st.write("Initial DataFrame columns:", df.columns.tolist())

# Data Preprocessing and Feature Engineering
def preprocess_data(df):
    # Ensure 'customerID' is dropped if it exists
    if 'customerID' in df.columns:
        df.drop(columns=['customerID'], inplace=True)
    
    # Convert 'TotalCharges' to numeric, handling errors
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    
    # Convert 'tenure' to integer
    df['tenure'] = df['tenure'].astype(int)
    
    # Define categorical columns
    categorical_columns = ['gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines',
                           'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
                           'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract',
                           'PaperlessBilling', 'PaymentMethod', 'Churn']
    
    # Check if all categorical columns exist
    missing_cols = [col for col in categorical_columns if col not in df.columns]
    if missing_cols:
        st.error(f"Missing columns: {missing_cols}")
        return df  # Return the DataFrame as is if columns are missing
    
    # Convert categorical columns to 'category' dtype
    df[categorical_columns] = df[categorical_columns].astype('category')
    
    # Fill missing 'TotalCharges' with median
    total_charges_median = df['TotalCharges'].median()
    df['TotalCharges'] = df['TotalCharges'].fillna(total_charges_median)
    
    # Fill missing values in categorical columns with mode
    for col in categorical_columns:
        if df[col].isnull().any():
            mode_value = df[col].mode()[0]
            df[col] = df[col].fillna(mode_value)
    
    # Create a new feature for phone service and multiple lines
    df['PhoneService_MultipleLines'] = df.apply(
        lambda row: 'No Phone Service' if row['PhoneService'] == 'No' else row['MultipleLines'], axis=1
    )
    
    # Calculate total services
    services = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']
    df['TotalServices'] = df[services].apply(lambda row: (row == 'Yes').sum(), axis=1)

    # Create tenure category
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
    
    # Calculate average monthly spend
    df['AverageMonthlySpend'] = df.apply(
        lambda row: row['TotalCharges'] / row['tenure'] if row['tenure'] > 0 else 0, axis=1
    )
    
    # Drop unnecessary columns
    df.drop(columns=['PhoneService', 'MultipleLines'] + services, inplace=True, errors='ignore')
    
    # Map 'Yes'/'No' to 1/0 for binary columns
    yes_no_columns = ['Partner', 'Dependents', 'PaperlessBilling', 'Churn']
    for col in yes_no_columns:
        df[col] = df[col].map({'Yes': 1, 'No': 0})
        mode_value = df[col].mode()[0]
        df[col] = df[col].fillna(mode_value).astype(int)
    
    # Fill missing values in numerical columns with median
    numerical_columns = ['MonthlyCharges', 'TotalCharges', 'AverageMonthlySpend']
    for col in numerical_columns:
        df[col].fillna(df[col].median(), inplace=True)

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

# Customer Profile Analysis
st.subheader("Customer Profile Analysis")
st.write("Demographic breakdown of customers:")
demographics = df.groupby('gender')['Churn'].value_counts(normalize=True).unstack().fillna(0)
st.bar_chart(demographics)

# Real-Time Churn Prediction
st.subheader("Real-Time Churn Prediction")
st.write("Enter customer details to predict churn risk:")
gender_input = st.selectbox("Gender", options=['Male', 'Female'])
partner_input = st.selectbox("Partner", options=['Yes', 'No'])
dependents_input = st.selectbox("Dependents", options=['Yes', 'No'])
tenure_input = st.number_input("Tenure (months)", min_value=0, max_value=72)
monthly_charges_input = st.number_input("Monthly Charges", min_value=0.0, format="%.2f")
total_charges_input = st.number_input("Total Charges", min_value=0.0, format="%.2f")

# Prepare input for prediction
input_data = {
    'gender': gender_input,
    'Partner': partner_input,
    'Dependents': dependents_input,
    'tenure': tenure_input,
    'MonthlyCharges': monthly_charges_input,
    'TotalCharges': total_charges_input
}
input_df = pd.DataFrame([input_data])

# Preprocess input data
input_df = preprocess_data(input_df)

# Predict churn
if st.button("Predict Churn"):
    prediction = log_reg_model.predict(input_df)
    st.write("Churn Prediction: ", "Yes" if prediction[0] == 1 else "No")

# Service Usage Trends
st.subheader("Service Usage Trends")
service_usage = df.groupby('TotalServices')['Churn'].mean().reset_index()
st.line_chart(service_usage.set_index('TotalServices'))

# Churn Risk Assessment
st.subheader("Churn Risk Assessment")
st.write("Assessing churn risk based on service usage:")
risk_assessment = df.groupby('TotalServices')['Churn'].mean().reset_index()
st.bar_chart(risk_assessment.set_index('TotalServices'))

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

def evaluate_model(model, X_train, X_test, y_train, y_test, model_name):
    model.fit(X_train, y_train)
    y_pred = model .predict(X_test)
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
    return model

log_reg = LogisticRegression(random_state=42)
rf = RandomForestClassifier(random_state=42)

st.subheader("Logistic Regression Evaluation")
log_reg_model = evaluate_model(log_reg, X_train, X_test, y_train, y_test, "Logistic Regression")

st.subheader("Random Forest Evaluation")
rf_model = evaluate_model(rf, X_train, X_test, y_train, y_test, "Random Forest") 

# Conclusion
st.subheader("Conclusion")
st.write("This application provides insights into customer churn, allowing for better decision-making and targeted strategies to retain customers. The interactive features enable users to explore data and make predictions based on individual customer profiles.")
