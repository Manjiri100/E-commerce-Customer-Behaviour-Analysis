import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report
from imblearn.over_sampling import SMOTE

# -----------------------
# Load Data
# -----------------------
df = pd.read_csv("retail_uk.csv", parse_dates=['InvoiceDate'])
print("Columns in data:", df.columns)
print("\nSample data:\n", df.head())

# -----------------------
# RFM Calculation
# -----------------------
snapshot_date = df['InvoiceDate'].max() + pd.Timedelta(days=1)

rfm = df.groupby('CustomerID').agg({
    'InvoiceDate': lambda x: (snapshot_date - x.max()).days,  # Recency
    'InvoiceNo': 'nunique',                                   # Frequency
    'Quantity': 'sum'                                         # Monetary (simplified)
}).reset_index()

rfm.columns = ['CustomerID', 'Recency', 'Frequency', 'Monetary']

# -----------------------
# Realistic Churn Labeling (~25% churn)
# -----------------------
np.random.seed(42)

# Base churn: Recency > 90 days
base_churn = (rfm['Recency'] > 90).astype(int)

# Random churn: 15% of active customers
random_churn = (np.random.rand(len(rfm)) < 0.15).astype(int)

# Combine
rfm['Churn'] = np.maximum(base_churn, random_churn)

print("\nClass distribution after labeling:\n", rfm['Churn'].value_counts())

# -----------------------
# Features & Target
# -----------------------
X = rfm[['Recency', 'Frequency', 'Monetary']]
y = rfm['Churn']

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# -----------------------
# Train/Test Split
# -----------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.3, random_state=42, stratify=y
)

print("\nClass distribution in training set before SMOTE: \n", pd.Series(y_train).value_counts())

# -----------------------
# SMOTE Oversampling
# -----------------------
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

print("\nClass distribution after SMOTE:\n", pd.Series(y_train_res).value_counts())

# -----------------------
# Logistic Regression
# -----------------------
lr = LogisticRegression(random_state=42)
lr.fit(X_train_res, y_train_res)
y_pred_lr = lr.predict(X_test)

print("\n--- Logistic Regression ---")
print("Accuracy:", lr.score(X_test, y_test))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_lr))
print("Classification Report:\n", classification_report(y_test, y_pred_lr))

# -----------------------
# Random Forest
# -----------------------
rf = RandomForestClassifier(random_state=42, n_estimators=100)
rf.fit(X_train_res, y_train_res)
y_pred_rf = rf.predict(X_test)

print("\n--- Random Forest ---")
print("Accuracy:", rf.score(X_test, y_test))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_rf))
print("Classification Report:\n", classification_report(y_test, y_pred_rf))
