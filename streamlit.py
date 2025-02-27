import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier


# Load the dataset
url = "WA_Fn-UseC_-Telco-Customer-Churn.csv"
df = pd.read_csv(url)

# Display the first 5 rows of the dataset
st.write("Initial Dataset:")
st.dataframe(df.head())

# Step 1: Remove unnecessary columns
df.drop(columns=['customerID'], inplace=True)

# Step 2: Change column data types
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df['tenure'] = df['tenure'].astype(int)
categorical_columns = ['gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines',
                       'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
                       'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract',
                       'PaperlessBilling', 'PaymentMethod', 'Churn']
df[categorical_columns] = df[categorical_columns].astype('category')

# Step 3: Handle missing values
total_charges_median = df['TotalCharges'].median()
df['TotalCharges'] = df['TotalCharges'].fillna(total_charges_median)

# Replace NaN in categorical columns with the mode
for col in categorical_columns:
    if df[col].isnull().any():
        mode_value = df[col].mode()[0]
        df[col] = df[col].fillna(mode_value)

# Step 4: Feature Engineering
df['PhoneService_MultipleLines'] = df.apply(
    lambda row: 'No Phone Service' if row['PhoneService'] == 'No' else row['MultipleLines'], axis=1
)

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

# Step 5: Visualizations

# Plot 1: Churn Distribution
fig, ax1 = plt.subplots(figsize=(8, 6))
sns.countplot(x='Churn', data=df, ax=ax1)
plt.title('Churn Distribution')
plt.xlabel('Churn (Yes/No)')
plt.ylabel('Count')

# Add total counts on top of the bars
for p in ax1.patches:
    ax1.annotate(f'{int(p.get_height())}', (p.get_x() + p.get_width() / 2., p.get_height()),
                 ha='center', va='center', fontsize=10, color='black', xytext=(0, 5),
                 textcoords='offset points')

plt.tight_layout()
st.pyplot(fig)


# Plot 2: Churn by Tenure Category
fig, ax2 = plt.subplots(figsize=(10, 6))
sns.countplot(x='TenureCategory', hue='Churn', data=df, ax=ax2)
plt.title('Churn by Tenure Category')
plt.xlabel('Tenure Category')
plt.ylabel('Count')

# Add total counts on top of the bars
for p in ax2.patches:
    ax2.annotate(f'{int(p.get_height())}', (p.get_x() + p.get_width() / 2., p.get_height()),
                 ha='center', va='center', fontsize=10, color='black', xytext=(0, 5),
                 textcoords='offset points')

plt.tight_layout()
st.pyplot(fig)


# Plot 3: Churn by Total Services
fig, ax3 = plt.subplots(figsize=(8, 6))
sns.boxplot(x='Churn', y='TotalServices', data=df, ax=ax3)
plt.title('Churn by Total Services')
plt.xlabel('Churn (Yes/No)')
plt.ylabel('Total Services')

plt.tight_layout()
st.pyplot(fig)


# Plot 4: Correlation Heatmap
df_encoded = pd.get_dummies(df, drop_first=True)
correlation_matrix = df_encoded.corr()
fig, ax = plt.subplots(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', ax=ax)
plt.title('Correlation Heatmap')
plt.tight_layout()
st.pyplot(fig)


# Step 6: Final Dataset
st.write("\nCleaned Dataset:")
st.dataframe(df.head())
st.write("\nDataset Info:")
st.write(df.info())


# List of columns with 'Yes'/'No' values
yes_no_columns = ['Partner', 'Dependents', 'PaperlessBilling', 'Churn']

# Replace 'Yes' with 1 and 'No' with 0 in the specified columns
for col in yes_no_columns:
    df[col] = df[col].map({'Yes': 1, 'No': 0})

    mode_value = df[col].mode()[0]
    df[col] = df[col].fillna(mode_value).astype(int)

# Replace NaN in numerical columns with the median
numerical_columns = ['MonthlyCharges', 'TotalCharges', 'AverageMonthlySpend']
for col in numerical_columns:
    median_value = df[col].median()
    df[col] = df[col].fillna(median_value)

# Convert 'PhoneService_MultipleLines' to binary
df['PhoneService_MultipleLines'] = df['PhoneService_MultipleLines'].apply(
    lambda x: 1 if x != 'No Phone Service' else 0
).astype(int)

# Display the updated dataset
st.write("\nUpdated Dataset:")
st.dataframe(df.head())
st.write("\nUpdated Dataset Info:")
st.write(df.info())


# Step 7: Train Models
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

# Evaluate Logistic Regression
log_reg_model = log_reg.fit(X_train, y_train)
y_pred_log_reg = log_reg.predict(X_test)
st.write("Logistic Regression Evaluation:")
st.write(classification_report(y_test, y_pred_log_reg))

# Evaluate Random Forest
rf_model = rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
st.write("Random Forest Evaluation:")
st.write(classification_report(y_test, y_pred_rf))


# Feature Importance (for Random Forest and XGBoost)
st.write("\nFeature Importance for Random Forest")
feat_importances = pd.Series(rf.feature_importances_, index=X_encoded.columns)
st.write(feat_importances.sort_values(ascending=False))

st.stop()
