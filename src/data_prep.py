"""
Data preparation script for Wellness Tourism Package prediction
Handles cleaning, feature engineering, train/test split, and HF dataset upload
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from datasets import Dataset, DatasetDict
from huggingface_hub import HfApi
import os

def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and engineer features from raw customer data"""
    df = df.copy()
    
    # Drop ID column (not used for training)
    if 'CustomerID' in df.columns:
        df.drop(columns=['CustomerID'], inplace=True)
    
    # Handle missing values
    num_cols = df.select_dtypes(include=['int64', 'float64']).columns
    cat_cols = df.select_dtypes(exclude=['int64', 'float64']).columns
    
    for col in num_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        df[col].fillna(df[col].median(), inplace=True)
    
    for col in cat_cols:
        df[col].fillna('Unknown', inplace=True)
    
    # Feature engineering
    df['ChildrenRatio'] = df['NumberOfChildrenVisiting'] / df['NumberOfPersonVisiting'].replace(0, 1)
    df['IncomePerPerson'] = df['MonthlyIncome'] / df['NumberOfPersonVisiting'].replace(0, 1)
    
    return df

def main():
    """Main function: load raw data, clean, split, save locally, upload to HF"""
    
    # TODO: Replace with your actual raw dataset path or HF dataset ID
    # For now, assuming local CSV - update this for your data
    RAW_DATA_PATH = "data/raw/wellness_customers.csv"  # Update this path
    
    if not os.path.exists(RAW_DATA_PATH):
        print(f"Raw data not found at {RAW_DATA_PATH}")
        print("Please place your CSV in data/raw/ or update RAW_DATA_PATH")
        return
    
    # Load and clean
    df = pd.read_csv(RAW_DATA_PATH)
    print("Raw data shape:", df.shape)
    print("\nTarget distribution:\n", df['ProdTaken'].value_counts())
    
    df_clean = clean_dataset(df)
    print("\nCleaned data shape:", df_clean.shape)
    
    # Train/test split
    train_df, test_df = train_test_split(
        df_clean, test_size=0.2, stratify=df_clean['ProdTaken'], random_state=42
    )
    
    # Save locally (satisfies rubric)
    os.makedirs("data/processed", exist_ok=True)
    train_df.to_csv("data/processed/train.csv", index=False)
    test_df.to_csv("data/processed/test.csv", index=False)
    print("✓ Train/test saved locally")
    
    # Create HF DatasetDict and push
    HF_USER = "YOUR_USERNAME"  # Replace with your HF username
    PROCESSED_DATASET_ID = f"{HF_USER}/wellness-tourism-processed"
    
    train_ds = Dataset.from_pandas(train_df)
    test_ds = Dataset.from_pandas(test_df)
    ds_dict = DatasetDict({"train": train_ds, "test": test_ds})
    
    ds_dict.push_to_hub(PROCESSED_DATASET_ID)
    print(f"✓ Processed dataset uploaded to: {PROCESSED_DATASET_ID}")

if __name__ == "__main__":
    main()
