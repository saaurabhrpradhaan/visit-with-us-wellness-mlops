"""
Model training script with hyperparameter tuning and HF model registration
Uses RandomForest with GridSearchCV and logs experiments
"""

import pandas as pd
import os
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
import skops.io as sio
from huggingface_hub import HfApi, create_repo
import csv

def log_experiment(model_name, params, roc_auc):
    """Log experiment results to CSV"""
    os.makedirs("experiments", exist_ok=True)
    filepath = "experiments/experiments_log.csv"
    
    with open(filepath, "a", newline="", encoding='utf-8') as f:
        writer = csv.DictWriter(
            f, fieldnames=["model", "n_estimators", "max_depth", "min_samples_split", "roc_auc"]
        )
        if f.tell() == 0:
            writer.writeheader()
        writer.writerow({
            "model": model_name,
            **params,
            "roc_auc": roc_auc
        })
    print(f"✓ Logged experiment to {filepath}")

def main():
    """Main training pipeline"""
    
    # Load processed data from local files
    train_df = pd.read_csv("data/processed/train.csv")
    test_df = pd.read_csv("data/processed/test.csv")
    
    X_train = train_df.drop(columns=["ProdTaken"])
    y_train = train_df["ProdTaken"]
    X_test = test_df.drop(columns=["ProdTaken"])
    y_test = test_df["ProdTaken"]
    
    # Identify column types
    num_cols = X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()
    cat_cols = X_train.select_dtypes(exclude=['int64', 'float64']).columns.tolist()
    
    print(f"Numeric features: {num_cols}")
    print(f"Categorical features: {cat_cols}")
    
    # Preprocessing pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", "passthrough", num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
        ]
    )
    
    # Random Forest pipeline
    rf = RandomForestClassifier(class_weight="balanced", random_state=42, n_jobs=-1)
    pipeline = Pipeline([("preprocessor", preprocessor), ("rf", rf)])
    
    # Hyperparameter grid
    param_grid = {
        "rf__n_estimators": [100, 200],
        "rf__max_depth": [None, 10],
        "rf__min_samples_split": [2, 5]
    }
    
    # Grid search
    grid_search = GridSearchCV(
        pipeline, param_grid, cv=3, scoring="roc_auc", n_jobs=-1, verbose=1
    )
    grid_search.fit(X_train, y_train)
    
    # Best model evaluation
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)
    y_proba = best_model.predict_proba(X_test)[:, 1]
    
    roc_auc = roc_auc_score(y_test, y_proba)
    print("\n=== BEST MODEL RESULTS ===")
    print(f"Best params: {grid_search.best_params_}")
    print(f"ROC-AUC: {roc_auc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Log experiment
    log_experiment("RandomForest", grid_search.best_params_, roc_auc)
    
    # Save and register model to HF Hub
    HF_USER = "SaaurabhR"  # Replace with your HF username
    MODEL_REPO_ID = f"{HF_USER}/wellness-wtp-rf-model"
    
    create_repo(MODEL_REPO_ID, repo_type="model", exist_ok=True)
    os.makedirs("model", exist_ok=True)
    
    model_path = "model/wellness_rf.skops"
    sio.dump(best_model, model_path)
    
    api = HfApi()
    api.upload_file(
        path_or_fileobj=model_path,
        path_in_repo="wellness_rf.skops",
        repo_id=MODEL_REPO_ID,
        repo_type="model"
    )
    
    print(f"✓ Model registered at: https://huggingface.co/{MODEL_REPO_ID}")

if __name__ == "__main__":
    main()
