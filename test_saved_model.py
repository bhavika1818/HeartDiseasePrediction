"""
===============================================================================
Heart Disease Prediction System - Model Verification Script
===============================================================================
This script verifies the newly saved leakage-free pipeline model (models/heart_disease_model.pkl).

Workflow:
1. Loads models/heart_disease_model.pkl using joblib.
2. Verifies that the loaded object is a valid Scikit-Learn Pipeline.
3. Loads dataset/heart.csv and removes exact duplicate rows.
4. Performs identical stratified train/test split (80/20, random_state=42).
5. Extracts the first patient sample from the test set (X_test).
6. Runs inference through the loaded pipeline and prints:
   - Patient clinical measurements
   - Actual ground truth target label
   - Model predicted class
   - Model prediction probability for Heart Disease (Class 1)
   - Match confirmation (Correct / Incorrect)
===============================================================================
"""

from pathlib import Path
import joblib
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

# Define file paths dynamically relative to script location
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "heart_disease_model.pkl"
DATA_PATH = BASE_DIR / "dataset" / "heart.csv"


def main():
    print("=" * 70)
    print("STEP 1: LOADING AND VERIFYING SAVED MODEL")
    print("=" * 70)
    
    # 1. Check if model file exists and load using joblib
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")
        
    loaded_model = joblib.load(MODEL_PATH)
    print(f"Loaded object type: {type(loaded_model)}")
    
    # 2. Verify that loaded object is a Scikit-Learn Pipeline instance
    is_pipeline = isinstance(loaded_model, Pipeline)
    print(f"Is valid Scikit-Learn Pipeline: {is_pipeline}")
    
    if not is_pipeline:
        raise TypeError("The loaded model is NOT a Scikit-Learn Pipeline!")
        
    print("\nPipeline steps:")
    for step_name, step_obj in loaded_model.steps:
        print(f"  - [{step_name}]: {type(step_obj).__name__}")

    # 3. Load dataset and remove exact duplicate rows
    print("\n" + "=" * 70)
    print("STEP 2: LOADING DATASET AND SPLITTING")
    print("=" * 70)
    df_raw = pd.read_csv(DATA_PATH)
    df_clean = df_raw.drop_duplicates().copy().reset_index(drop=True)
    print(f"Clean dataset shape: {df_clean.shape}")

    # Separate features and target label
    X = df_clean.drop(columns=["target"])
    y = df_clean["target"]

    # 4. Perform identical Stratified 80/20 train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # 5. Extract the first patient row from X_test and y_test
    print("\n" + "=" * 70)
    print("STEP 3: EXTRACTING TEST SAMPLE & RUNNING PREDICTION")
    print("=" * 70)
    
    sample_features = X_test.iloc[[0]]  # Keep as 1-row DataFrame for pipeline input
    actual_target = int(y_test.iloc[0])

    # 6. Run prediction and probability estimation
    prediction = int(loaded_model.predict(sample_features)[0])
    probabilities = loaded_model.predict_proba(sample_features)[0]
    prob_class_1 = float(probabilities[1])  # Probability of Heart Disease (Class 1)

    # Print results formatted clearly for a beginner
    print("1. Patient Feature Values:")
    for col in sample_features.columns:
        print(f"   - {col:10s}: {sample_features.iloc[0][col]}")
        
    target_desc = "1 (Heart Disease Present)" if actual_target == 1 else "0 (No Heart Disease)"
    pred_desc = "1 (Heart Disease Present)" if prediction == 1 else "0 (No Heart Disease)"
    matches = (prediction == actual_target)

    print(f"\n2. Actual Target              : {target_desc}")
    print(f"3. Loaded Pipeline Prediction : {pred_desc}")
    print(f"4. Class 1 Probability        : {prob_class_1:.4f} ({prob_class_1 * 100:.2f}%)")
    
    print("\n" + "=" * 70)
    print("VERIFICATION RESULT:")
    print("=" * 70)
    if matches:
        print(f"SUCCESS: Pipeline prediction ({prediction}) matches actual target ({actual_target})!")
    else:
        print(f"NOTICE: Pipeline prediction ({prediction}) does not match actual target ({actual_target}).")


if __name__ == "__main__":
    main()
