import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score


def build_churn_dataset(rfm):
    """
    Create churn dataset (no training here)
    """

    df = rfm.copy()

    # churn definition (data-driven)
    threshold = df["Recency"].quantile(0.75)
    df["Churn"] = (df["Recency"] >= threshold).astype(int)

    X = df[["Recency", "Frequency", "Monetary"]]
    y = df["Churn"]

    return X, y


def train_churn_model(X, y):
    """
    Train churn model only
    """

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=6,
        random_state=42
    )

    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1]

    print("\n--- CHURN MODEL PERFORMANCE ---")
    print(classification_report(y_test, preds))
    print("ROC-AUC:", roc_auc_score(y_test, probs))

    # feature importance
    importance = pd.DataFrame({
        "feature": X.columns,
        "importance": model.feature_importances_
    }).sort_values(by="importance", ascending=False)

    print("\nFEATURE IMPORTANCE:\n", importance)

    return model

def run_churn_analysis(df):
    print("\n⚠️ STARTING CHURN ANALYSIS...\n")

    # Build dataset
    X, y = build_churn_dataset(df)

    # Train model
    model = train_churn_model(X, y)

    print("\n📊 CHURN MODEL TRAINED SUCCESSFULLY")

    return model