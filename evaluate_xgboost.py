import pandas as pd
import xgboost as xgb
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, classification_report,
)

FEATURE_COLUMNS = [
    "CreditScore",
    "Gender",
    "Age",
    "Tenure",
    "Balance",
    "NumOfProducts",
    "HasCrCard",
    "IsActiveMember",
]
TARGET_COLUMN = "Exited"

def evaluate(model_path: str, val_path: str):
    print(f"Loading data from {val_path}...")
    val_df = pd.read_csv(val_path)
    X_val = val_df[FEATURE_COLUMNS]
    y_val = val_df[TARGET_COLUMN]

    print(f"Loading model from {model_path}...")
    model = xgb.XGBClassifier()
    model.load_model(model_path)

    print("Making predictions...")
    y_pred = model.predict(X_val)
    y_proba = model.predict_proba(X_val)[:, 1]  # probability of churn (class 1)

    print("\n--- Model Metrics ---")
    print("Accuracy: ", accuracy_score(y_val, y_pred))
    print("Precision:", precision_score(y_val, y_pred))
    print("Recall:   ", recall_score(y_val, y_pred))
    print("F1-score: ", f1_score(y_val, y_pred))
    print("ROC-AUC:  ", roc_auc_score(y_val, y_proba))

    print("\nConfusion matrix (rows=actual, cols=predicted):")
    print(confusion_matrix(y_val, y_pred))

    print("\nFull classification report:")
    print(classification_report(y_val, y_pred, target_names=["Stayed", "Churned"]))


if __name__ == "__main__":
    evaluate(model_path="xgboost_churn_model.json", val_path="./splits/val.csv")