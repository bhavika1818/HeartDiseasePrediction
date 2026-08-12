"""
===============================================================================
Heart Disease Prediction System - Data Integrity & Preprocessing Workflow
===============================================================================
Phase 1: Data Integrity, Deduplication, Anomaly Analysis & Leakage Prevention

This script implements a reproducible data-preparation workflow:
1. Loads the original raw dataset (without modifying dataset/heart.csv).
2. Performs comprehensive data audit (shape, duplicates, missing values, target balance).
3. Deduplicates dataset to eliminate duplicate leakage across train/test splits.
4. Documents and handles thal=0 anomalies cleanly.
5. Performs Stratified Train/Test split AFTER deduplication.
6. Builds a Scikit-Learn ColumnTransformer pipeline (StandardScaler + OneHotEncoder).
7. Fits preprocessor ONLY on the training split to prevent statistical data leakage.
===============================================================================
"""

from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer


def load_and_audit_data(data_path: Path):
    """
    Loads raw CSV dataset and performs initial integrity inspection.
    WHY: Auditing raw data ensures transparency before any ML modifications.
    """
    print("=" * 70)
    print("STEP 1: RAW DATASET INTEGRITY AUDIT")
    print("=" * 70)
    
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found at {data_path}")
        
    df_raw = pd.read_csv(data_path)
    
    raw_rows, raw_cols = df_raw.shape
    duplicate_count = df_raw.duplicated().sum()
    null_count = df_raw.isnull().sum().sum()
    unique_rows = raw_rows - duplicate_count
    
    print(f"Original Row Count   : {raw_rows}")
    print(f"Original Column Count: {raw_cols}")
    print(f"Missing Values (NaN) : {null_count}")
    print(f"Duplicate Rows Count : {duplicate_count} ({duplicate_count / raw_rows * 100:.1f}%)")
    print(f"Unique Patient Rows  : {unique_rows}")
    
    print("\nRaw Class Distribution (target):")
    raw_target_counts = df_raw['target'].value_counts()
    for label, count in raw_target_counts.items():
        desc = "Heart Disease Present" if label == 1 else "No Disease"
        pct = count / raw_rows * 100
        print(f"  Class {label} ({desc}): {count} samples ({pct:.2f}%)")
        
    return df_raw


def deduplicate_dataset(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Removes exact duplicate patient records from the dataset.
    WHY: 723 rows in heart.csv are exact duplicates of original 302 unique patients.
    If random train_test_split is performed on the raw dataset, identical patient 
    records appear in both X_train and X_test, causing severe Data Leakage and 
    inflating test accuracy to ~98-100% spuriously. Deduplicating first ensures 
    the model is evaluated on genuinely unseen patient data.
    """
    print("\n" + "=" * 70)
    print("STEP 2: DATA DEDUPLICATION")
    print("=" * 70)
    
    df_clean = df_raw.drop_duplicates().copy().reset_index(drop=True)
    
    print(f"Rows before deduplication: {len(df_raw)}")
    print(f"Rows after deduplication : {len(df_clean)}")
    print(f"Duplicate rows removed   : {len(df_raw) - len(df_clean)}")
    
    print("\nDeduplicated Class Distribution (target):")
    clean_target_counts = df_clean['target'].value_counts()
    for label, count in clean_target_counts.items():
        desc = "Heart Disease Present" if label == 1 else "No Disease"
        pct = count / len(df_clean) * 100
        print(f"  Class {label} ({desc}): {count} samples ({pct:.2f}%)")
        
    return df_clean


def investigate_thal_zero(df_clean: pd.DataFrame):
    """
    Investigates and documents thal=0 records.
    WHY: In standard UCI Cleveland documentation, thal valid codes are:
      1 = Normal
      2 = Fixed defect
      3 = Reversible defect
    Value 0 is an invalid/missing value code in legacy data collection.
    Rather than silently deleting patient records or imputing without reason,
    we explicitly preserve 0 as a distinct 'Unknown / Other' category code.
    This matches index.html option: <option value="0">Unknown / other</option>.
    """
    print("\n" + "=" * 70)
    print("STEP 3: THAL=0 ANOMALY INVESTIGATION")
    print("=" * 70)
    
    thal_zero_rows = df_clean[df_clean['thal'] == 0]
    count_thal_zero = len(thal_zero_rows)
    
    print(f"Unique records with thal = 0: {count_thal_zero}")
    if count_thal_zero > 0:
        print("\nDetails of thal=0 unique records:")
        for idx, row in thal_zero_rows.iterrows():
            sex_str = "Male" if row['sex'] == 1 else "Female"
            target_str = "Disease" if row['target'] == 1 else "No Disease"
            print(f"  Patient ID {idx}: Age={row['age']}, Sex={sex_str}, CP={row['cp']}, "
                  f"BP={row['trestbps']}, Chol={row['chol']}, Target={target_str}")
                  
    print("\nHandling Strategy for thal=0:")
    print("  -> Preserved as categorical value '0' ('Unknown / Other').")
    print("  -> When One-Hot Encoded, thal=0 receives its own dummy indicator column.")
    print("  -> Prevents data loss while eliminating false ordinal assumptions (0 < 1 < 2 < 3).")


def split_data_leakage_free(df_clean: pd.DataFrame, test_size=0.2, random_state=42):
    """
    Performs Stratified Train/Test split AFTER deduplication.
    WHY: 
    1. Stratified split maintains identical target class ratios in train and test sets.
    2. Splitting AFTER deduplication guarantees zero data overlap between train and test.
    3. Reproducible split using fixed random_state=42.
    """
    print("\n" + "=" * 70)
    print("STEP 4: STRATIFIED TRAIN/TEST SPLIT (LEAKAGE PREVENTED)")
    print("=" * 70)
    
    X = df_clean.drop(columns=['target'])
    y = df_clean['target']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        stratify=y,
        random_state=random_state
    )
    
    print(f"Training Set Shape: X_train={X_train.shape}, y_train={y_train.shape}")
    print(f"Testing Set Shape : X_test={X_test.shape}, y_test={y_test.shape}")
    
    print("\nTraining Set Class Proportions:")
    train_pct = y_train.value_counts(normalize=True) * 100
    for label, pct in train_pct.items():
        print(f"  Class {label}: {pct:.2f}%")
        
    print("Testing Set Class Proportions:")
    test_pct = y_test.value_counts(normalize=True) * 100
    for label, pct in test_pct.items():
        print(f"  Class {label}: {pct:.2f}%")
        
    return X_train, X_test, y_train, y_test


def build_and_fit_preprocessor(X_train: pd.DataFrame, X_test: pd.DataFrame):
    """
    Constructs and fits Scikit-Learn ColumnTransformer preprocessor.
    WHY: 
    1. Numerical features must be standardized (StandardScaler) for distance models.
    2. Categorical features must be One-Hot Encoded to avoid artificial ordering assumptions.
    3. Preprocessor MUST be fitted strictly on X_train (fit_transform) and only 
       applied to X_test (transform) to prevent Statistical Data Leakage.
    """
    print("\n" + "=" * 70)
    print("STEP 5: PREPROCESSING PIPELINE FITTING (TRAIN-ONLY FIT)")
    print("=" * 70)
    
    num_features = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
    cat_features = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), num_features),
            ('cat', OneHotEncoder(sparse_output=False, handle_unknown='ignore'), cat_features)
        ],
        remainder='passthrough'
    )
    
    # Fit ONLY on training data
    X_train_processed = preprocessor.fit_transform(X_train)
    # Transform test data using training statistics
    X_test_processed = preprocessor.transform(X_test)
    
    print("Numerical Features   :", num_features)
    print("Categorical Features :", cat_features)
    print(f"Transformed X_train shape: {X_train_processed.shape} (Expanded via One-Hot Encoding)")
    print(f"Transformed X_test shape : {X_test_processed.shape}")
    print("\nPreprocessor fitted strictly on X_train. Statistical parameters (mean, std) learned exclusively from training set.")
    
    return preprocessor, X_train_processed, X_test_processed


def generate_data_quality_report(df_raw: pd.DataFrame, df_clean: pd.DataFrame, X_train: pd.DataFrame, X_test: pd.DataFrame):
    """
    Prints final structured summary report.
    """
    print("\n" + "=" * 70)
    print("FINAL DATA QUALITY & INTEGRITY REPORT")
    print("=" * 70)
    
    raw_rows = len(df_raw)
    dup_rows = df_raw.duplicated().sum()
    unique_rows = len(df_clean)
    null_count = df_raw.isnull().sum().sum()
    thal_zero_count = (df_clean['thal'] == 0).sum()
    
    report_data = {
        "Metric": [
            "Original Dataset Rows",
            "Duplicate Rows Identified & Removed",
            "Final Clean Unique Rows",
            "Missing Values (NaN)",
            "Target Class 1 (Disease Present)",
            "Target Class 0 (No Disease)",
            "thal=0 Anomaly Records (Unique)",
            "Training Subset Size (80%)",
            "Testing Subset Size (20%)",
            "Reproducible Seed (random_state)"
        ],
        "Value": [
            f"{raw_rows}",
            f"{dup_rows} ({dup_rows/raw_rows*100:.1f}%)",
            f"{unique_rows}",
            f"{null_count}",
            f"{df_clean['target'].sum()} ({df_clean['target'].mean()*100:.1f}%)",
            f"{(df_clean['target']==0).sum()} ({(1-df_clean['target'].mean())*100:.1f}%)",
            f"{thal_zero_count} (Preserved as Unknown category)",
            f"{len(X_train)} samples",
            f"{len(X_test)} samples",
            "42"
        ]
    }
    
    report_df = pd.DataFrame(report_data)
    print(report_df.to_string(index=False))
    print("=" * 70)


def main():
    base_dir = Path(__file__).resolve().parent
    data_path = base_dir / "dataset" / "heart.csv"
    
    # 1. Audit raw dataset
    df_raw = load_and_audit_data(data_path)
    
    # 2. Deduplicate dataset
    df_clean = deduplicate_dataset(df_raw)
    
    # 3. Investigate thal=0 values
    investigate_thal_zero(df_clean)
    
    # 4. Stratified Train/Test split
    X_train, X_test, y_train, y_test = split_data_leakage_free(df_clean)
    
    # 5. Build preprocessor & fit strictly on train split
    preprocessor, X_train_proc, X_test_proc = build_and_fit_preprocessor(X_train, X_test)
    
    # 6. Print final Data Quality Report
    generate_data_quality_report(df_raw, df_clean, X_train, X_test)


if __name__ == "__main__":
    main()
