import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score
)

from imblearn.over_sampling import SMOTE

# -----------------------
# 1. Load Data
# -----------------------
df = pd.read_csv("retail_uk.csv", parse_dates=["InvoiceDate"])

# Basic cleanup (important in real projects)
df = df.dropna(subset=["CustomerID"])
df["CustomerID"] = df["CustomerID"].astype(int)

# -----------------------
# 2. Feature Engineering (RFM)
# -----------------------
snapshot_date = df["InvoiceDate"].max() + pd.Timedelta(days=1)

rfm = df.groupby("CustomerID").agg(
    Recency=("InvoiceDate", lambda x: (snapshot_date - x.max()).days),
    Frequency=("InvoiceNo", "nunique"),
    Monetary=("Quantity", "sum")  # replace with Quantity * Price if available
).reset_index()

# Optional improvement (recommended if UnitPrice exists):
# df["Revenue"] = df["Quantity"] * df["UnitPrice"]

# -----------------------
# 3. Churn Definition (Business-Defensible)
# -----------------------
# Definition: customer is churned if inactive for > 90 days
CHURN_THRESHOLD = 90
rfm["Churn"] = (rfm["Recency"] > CHURN_THRESHOLD).astype(int)

print("Churn distribution:")
print(rfm["Churn"].value_counts(normalize=True))

# -----------------------
# 4. Features & Target
# -----------------------
X = rfm[["Recency", "Frequency", "Monetary"]]
y = rfm["Churn"]

# -----------------------
# 5. Train/Test Split (NO leakage)
# -----------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.3,
    random_state=42,
    stratify=y
)

# -----------------------
# 6. Scaling (fit ONLY on train)
# -----------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# -----------------------
# 7. Handle Class Imbalance (SMOTE only on training data)
# -----------------------
smote = SMOTE(random_state=42)

X_train_res, y_train_res = smote.fit_resample(
    X_train_scaled,
    y_train
)

print("\nAfter SMOTE:")
print(pd.Series(y_train_res).value_counts())

# -----------------------
# 8. Model 1: Logistic Regression (Baseline)
# -----------------------
lr = LogisticRegression(max_iter=1000, random_state=42)
lr.fit(X_train_res, y_train_res)

lr_pred = lr.predict(X_test_scaled)
lr_proba = lr.predict_proba(X_test_scaled)[:, 1]

print("\n====================")
print("LOGISTIC REGRESSION")
print("====================")
print("ROC-AUC:", roc_auc_score(y_test, lr_proba))
print(confusion_matrix(y_test, lr_pred))
print(classification_report(y_test, lr_pred))

# -----------------------
# 9. Model 2: Random Forest (Non-linear model)
# -----------------------
rf = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    class_weight="balanced"
)

rf.fit(X_train_res, y_train_res)

rf_pred = rf.predict(X_test_scaled)
rf_proba = rf.predict_proba(X_test_scaled)[:, 1]

print("\n====================")
print("RANDOM FOREST")
print("====================")
print("ROC-AUC:", roc_auc_score(y_test, rf_proba))
print(confusion_matrix(y_test, rf_pred))
print(classification_report(y_test, rf_pred))

# -----------------------
# 10. Feature Importance (Business Insight Layer)
# -----------------------
feature_importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": rf.feature_importances_
}).sort_values(by="Importance", ascending=False)

print("\n====================")
print("FEATURE IMPORTANCE")
print("====================")
print(feature_importance)

# -----------------------
# 11. Business Interpretation (Key for Interviews)
# -----------------------
print("\n====================")
print("BUSINESS INSIGHT SUMMARY")
print("====================")

top_feature = feature_importance.iloc[0]

print(
    f"Most important churn driver: {top_feature['Feature']} "
    f"(importance = {top_feature['Importance']:.3f})"
)

print(
    "Recommendation: Target high-risk customers with high recency and low frequency "
    "using retention campaigns or incentives."
)
