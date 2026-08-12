"""
===============================================================================
Heart Disease Prediction System - Final Pipeline Training Script
===============================================================================
This script builds a leakage-free Scikit-Learn Pipeline combining data preprocessing
(StandardScaler + OneHotEncoder) and a Logistic Regression classifier.

Workflow:
1. Loads dataset/heart.csv
2. Removes exact duplicate patient rows (deduplication)
3. Separates 13 clinical features from the target column ('target')
4. Performs a stratified 80/20 train/test split (random_state=42)
5. Builds a ColumnTransformer for numerical and categorical features
6. Constructs a Pipeline (preprocessor + LogisticRegression)
7. Fits the Pipeline ONLY on X_train and y_train
8. Evaluates performance on the untouched X_test set
9. Saves the COMPLETE fitted Pipeline object to disk for Flask app deployment
===============================================================================
"""

from pathlib import Path
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)

# Step 1: Define file paths dynamically relative to script location
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "dataset" / "heart.csv"
MODEL_PATH_ROOT = BASE_DIR / "heart_disease_model.pkl"
MODEL_PATH_MODELS = BASE_DIR / "models" / "heart_disease_model.pkl"


def main():
    print("=" * 70)
    print("STEP 1: LOADING DATASET")
    print("=" * 70)
    # Load raw dataset from dataset/heart.csv
    df_raw = pd.read_csv(DATA_PATH)
    print(f"Raw dataset shape: {df_raw.shape}")

    # Step 2: Remove exact duplicate patient rows to prevent data leakage
    print("\n" + "=" * 70)
    print("STEP 2: REMOVING DUPLICATES (LEAKAGE PREVENTION)")
    print("=" * 70)
    df_clean = df_raw.drop_duplicates().copy().reset_index(drop=True)
    print(f"Clean dataset shape: {df_clean.shape} (Removed {len(df_raw) - len(df_clean)} duplicate rows)")

    # Step 3: Separate 13 clinical features (X) from target output label (y)
    X = df_clean.drop(columns=["target"])
    y = df_clean["target"]

    # Step 4: Perform 80/20 Stratified Train/Test split
    print("\n" + "=" * 70)
    print("STEP 3: STRATIFIED TRAIN/TEST SPLIT (80/20)")
    print("=" * 70)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    print(f"Training set size : {X_train.shape[0]} samples")
    print(f"Test set size     : {X_test.shape[0]} samples")

    # Step 5 & 6: Explicitly group numerical and categorical feature column names
    num_features = ["age", "trestbps", "chol", "thalach", "oldpeak"]
    cat_features = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]

    # Step 7, 8 & 9: Construct ColumnTransformer
    # Numerical features -> StandardScaler (centers data to mean=0, std=1)
    # Categorical features -> OneHotEncoder (converts discrete category integers into binary dummy variables)
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_features),
        ],
        remainder="passthrough"
    )

    # Step 10: Construct full Scikit-Learn Pipeline combining preprocessor & classifier
    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", LogisticRegression(max_iter=1000, random_state=42)),
        ]
    )

    # Step 11: Fit pipeline ONLY on training data (X_train, y_train) to prevent statistical leakage
    print("\n" + "=" * 70)
    print("STEP 4: FITTING PIPELINE ON TRAINING DATA ONLY")
    print("=" * 70)
    pipeline.fit(X_train, y_train)
    print("Pipeline fitted successfully on X_train and y_train.")

    # Step 12: Evaluate pipeline on untouched X_test set
    print("\n" + "=" * 70)
    print("STEP 5: EVALUATING PIPELINE ON UNTOUCHED TEST DATA")
    print("=" * 70)
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    # Step 13: Compute and display detailed evaluation metrics
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    auc = roc_auc_score(y_test, y_proba)
    cm = confusion_matrix(y_test, y_pred)

    print(f"Accuracy       : {acc:.4f} ({acc * 100:.2f}%)")
    print(f"Precision      : {prec:.4f}")
    print(f"Recall         : {rec:.4f}")
    print(f"F1-Score       : {f1:.4f}")
    print(f"ROC-AUC        : {auc:.4f}")
    print("\nConfusion Matrix:")
    print(cm)
    tn, fp, fn, tp = cm.ravel()
    print(f"  True Negatives (TN) : {tn}")
    print(f"  False Positives (FP): {fp}")
    print(f"  False Negatives (FN): {fn}")
    print(f"  True Positives (TP) : {tp}")

    # Step 14: Save the COMPLETE fitted Pipeline object to disk
    print("\n" + "=" * 70)
    print("STEP 6: SAVING COMPLETE FITTED PIPELINE")
    print("=" * 70)
    
    # Save to models/heart_disease_model.pkl
    MODEL_PATH_MODELS.parent.mkdir(exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH_MODELS)
    print(f"Saved complete Pipeline to: {MODEL_PATH_MODELS}")

    # Save to root heart_disease_model.pkl for Flask app (app.py) loading
    joblib.dump(pipeline, MODEL_PATH_ROOT)
    print(f"Saved complete Pipeline to: {MODEL_PATH_ROOT}")
    print("=" * 70)
    print("Pipeline creation and export complete!")


if __name__ == "__main__":
    main()
