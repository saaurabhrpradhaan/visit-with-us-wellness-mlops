# ========================================
# CREATE WORKING train.py (Complete replacement)
# ========================================
train_py_content = '''
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
import skops.io as sio
from huggingface_hub import HfApi, create_repo
import os

def main():
    print("Loading processed data...")
    train_df = pd.read_csv("data/processed/train.csv")
    test_df = pd.read_csv("data/processed/test.csv")
    
    X_train = train_df.drop(columns=["ProdTaken"])
    y_train = train_df["ProdTaken"]
    X_test = test_df.drop(columns=["ProdTaken"])
    y_test = test_df["ProdTaken"]
    
    print(f"Training data shape: {X_train.shape}")
    
    # Identify feature types
    num_features = X_train.select_dtypes(include=[np.number]).columns.tolist()
    cat_features = X_train.select_dtypes(exclude=[np.number]).columns.tolist()
    
    print(f"Numeric features ({len(num_features)}): {num_features}")
    print(f"Categorical features ({len(cat_features)}): {cat_features[:5]}...")
    
    # Preprocessing pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", "passthrough", num_features),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_features)
        ]
    )
    
    # Random Forest model
    rf = RandomForestClassifier(class_weight="balanced", random_state=42, n_jobs=-1)
    pipeline = Pipeline([("preprocessor", preprocessor), ("rf", rf)])
    
    # Hyperparameter tuning
    param_grid = {
        "rf__n_estimators": [100, 200],
        "rf__max_depth": [None, 10],
        "rf__min_samples_split": [2, 5]
    }
    
    print("Starting GridSearchCV...")
    grid_search = GridSearchCV(pipeline, param_grid, cv=3, scoring="roc_auc", n_jobs=-1)
    grid_search.fit(X_train, y_train)
    
    # Best model evaluation
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)
    y_proba = best_model.predict_proba(X_test)[:, 1]
    
    roc_auc = roc_auc_score(y_test, y_proba)
    
    print("\\n=== RESULTS ===")
    print(f"Best params: {grid_search.best_params_}")
    print(f"ROC-AUC: {roc_auc:.4f}")
    print("\\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Log experiment
    os.makedirs("experiments", exist_ok=True)
    experiment_df = pd.DataFrame([{
        "model": "RandomForest",
        "n_estimators": grid_search.best_params_["rf__n_estimators"],
        "max_depth": grid_search.best_params_["rf__max_depth"],
        "min_samples_split": grid_search.best_params_["rf__min_samples_split"],
        "roc_auc": roc_auc
    }])
    experiment_df.to_csv("experiments/experiments_log.csv", index=False)
    print("\\n✅ Experiment logged!")
    
    # Save and upload model
    HF_USER = "SaaurabhR"
    MODEL_REPO_ID = f"{HF_USER}/wellness-wtp-rf-model"
    
    print(f"\\nUploading model to HF Hub...")
    create_repo(MODEL_REPO_ID, repo_type="model", exist_ok=True)
    
    model_path = "model/wellness_rf.skops"
    sio.dump(best_model, model_path)
    
    api = HfApi()
    api.upload_file(
        path_or_fileobj=model_path,
        path_in_repo="wellness_rf.skops",
        repo_id=MODEL_REPO_ID,
        repo_type="model"
    )
    
    print(f"✅ Model uploaded: https://huggingface.co/{MODEL_REPO_ID}")

if __name__ == "__main__":
    main()
'''

# Write the fixed train.py
with open('src/train.py', 'w') as f:
    f.write(train_py_content)

print("✅ train.py FIXED & READY!")
print("Now run: !python src/train.py")
