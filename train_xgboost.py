import pandas as pd
import xgboost as xgb
import wandb
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
)
from sklearn.model_selection import GridSearchCV

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
RANDOM_STATE = 42

def train(train_path: str, val_path: str):
    # Load data
    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)

    X_train = train_df[FEATURE_COLUMNS]
    y_train = train_df[TARGET_COLUMN]

    X_val = val_df[FEATURE_COLUMNS]
    y_val = val_df[TARGET_COLUMN]

    # Calculate scale_pos_weight
    num_neg = (y_train == 0).sum()
    num_pos = (y_train == 1).sum()
    scale_pos_weight = num_neg / num_pos
    print(f"Calculated scale_pos_weight: {scale_pos_weight:.3f} (Negatives: {num_neg}, Positives: {num_pos})")

    # Hyperparameter Grid
    param_grid = {
        "n_estimators": [100, 200, 300],
        "max_depth": [3, 4, 5],
        "learning_rate": [0.05, 0.1, 0.2]
    }

    print("Starting GridSearchCV for Hyperparameter Tuning...")
    base_model = xgb.XGBClassifier(
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=RANDOM_STATE,
        scale_pos_weight=scale_pos_weight
    )

    # We use 'f1' scoring because the dataset is imbalanced
    grid_search = GridSearchCV(
        estimator=base_model, 
        param_grid=param_grid, 
        scoring='f1', 
        cv=3, 
        verbose=1
    )
    
    grid_search.fit(X_train, y_train)

    print("\nBest Parameters Found:")
    print(grid_search.best_params_)

    # Use the best model
    model = grid_search.best_estimator_

    # Initialize W&B
    wandb.init(project="churn-prediction", name="xgboost-tuned", config=grid_search.best_params_)

    print("\nEvaluating Best Model on Validation Set...")
    y_pred = model.predict(X_val)
    y_proba = model.predict_proba(X_val)[:, 1]

    # Calculate metrics
    accuracy = accuracy_score(y_val, y_pred)
    precision = precision_score(y_val, y_pred)
    recall = recall_score(y_val, y_pred)
    f1 = f1_score(y_val, y_pred)
    roc_auc = roc_auc_score(y_val, y_proba)
    
    metrics = {
        "val_accuracy": accuracy,
        "val_precision": precision,
        "val_recall": recall,
        "val_f1": f1,
        "val_roc_auc": roc_auc
    }
    
    print("\nValidation Metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.3f}")

    wandb.log(metrics)

    # Confusion matrix W&B plot
    wandb.log({"confusion_matrix": wandb.plot.confusion_matrix(
        probs=None,
        y_true=y_val.to_numpy(),
        preds=y_pred,
        class_names=["Stayed", "Churned"]
    )})

    # Log feature importances
    feature_importances = pd.DataFrame({
        "feature": FEATURE_COLUMNS,
        "importance": model.feature_importances_
    }).sort_values(by="importance", ascending=False)
    print("\nFeature Importances:")
    print(feature_importances)

    # Save the model
    model_path = "xgboost_churn_model.json"
    model.save_model(model_path)
    print(f"\nModel saved to {model_path}")
    
    # Finish W&B
    wandb.finish()


if __name__ == "__main__":
    train(train_path="./splits/train.csv", val_path="./splits/val.csv")
