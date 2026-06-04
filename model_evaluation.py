import os
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    accuracy_score
)

# -----------------------------
# Paths (robust)
# -----------------------------
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

MODELS_DIR = os.path.join(BASE_DIR, "models")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")

os.makedirs(REPORTS_DIR, exist_ok=True)


# -----------------------------
# Load model safely
# -----------------------------
def load_model(model_name: str):
    model_path = os.path.join(MODELS_DIR, model_name)

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")

    return joblib.load(model_path)


# -----------------------------
# Evaluate Classification Model
# -----------------------------
def evaluate_churn_model(model, X_test, y_test, save_report=True):
    print("\n--- MODEL EVALUATION (CHURN) ---")

    y_pred = model.predict(X_test)

    # Some models support predict_proba
    y_proba = None
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    cm = confusion_matrix(y_test, y_pred)

    roc_auc = None
    if y_proba is not None:
        roc_auc = roc_auc_score(y_test, y_proba)

    print(f"Accuracy: {acc:.4f}")
    if roc_auc:
        print(f"ROC-AUC: {roc_auc:.4f}")

    print("\nConfusion Matrix:")
    print(cm)

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    results = {
        "accuracy": float(acc),
        "roc_auc": float(roc_auc) if roc_auc else None,
        "confusion_matrix": cm.tolist(),
        "classification_report": report
    }

    if save_report:
        report_path = os.path.join(REPORTS_DIR, "model_evaluation.json")
        with open(report_path, "w") as f:
            json.dump(results, f, indent=4)

        print(f"\nSaved evaluation report → {report_path}")

    return results


# -----------------------------
# Feature Importance (Tree Models)
# -----------------------------
def get_feature_importance(model, feature_names):
    if not hasattr(model, "feature_importances_"):
        print("Feature importance not available for this model.")
        return None

    importance_df = pd.DataFrame({
        "feature": feature_names,
        "importance": model.feature_importances_
    }).sort_values(by="importance", ascending=False)

    return importance_df


# -----------------------------
# Example standalone run
# -----------------------------
if __name__ == "__main__":
    print("Loading model...")

    model = load_model("churn_model.pkl")

    print("Model loaded successfully.")

    # Dummy placeholder (replace with real test data in pipeline)
    print("NOTE: Pass X_test, y_test from pipeline for evaluation.")