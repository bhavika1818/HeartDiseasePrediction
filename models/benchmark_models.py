"""
===============================================================================
Phase 2A: Leakage-Free Multi-Model Benchmark
===============================================================================
This script evaluates the 5 baseline classification algorithms already present in 
the project using strict leakage-free pipelines and Phase 1 data cleaning:

Algorithms Evaluated:
1. Logistic Regression
2. Decision Tree Classifier
3. Random Forest Classifier
4. K-Nearest Neighbors (KNN)
5. Support Vector Machine (SVM)

Workflow:
1. Load dataset/heart.csv and deduplicate to 302 unique patient records.
2. Perform reproducible Stratified 80/20 train/test split (241 train / 61 test).
3. Construct sklearn Pipeline for each model (StandardScaler + OneHotEncoder + Model).
4. Run Stratified 5-Fold Cross-Validation on X_train/y_train (CV Accuracy & CV ROC-AUC).
5. Fit pipeline on full X_train and evaluate once on untouched X_test.
6. Calculate Test Accuracy, Precision, Recall, F1-Score, and Test ROC-AUC.
7. Print Confusion Matrices and plot ROC Curve comparisons (saved to models/roc_comparison.png).
8. Print sorted comparison table.
===============================================================================
"""

from pathlib import Path
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for headless execution
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, roc_curve
)


def load_and_preprocess_data(data_path: Path):
    """Loads dataset/heart.csv and applies Phase 1 deduplication and stratified split."""
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found at {data_path}")
        
    df_raw = pd.read_csv(data_path)
    df_clean = df_raw.drop_duplicates().copy().reset_index(drop=True)
    
    X = df_clean.drop(columns=['target'])
    y = df_clean['target']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    
    return X_train, X_test, y_train, y_test, df_clean


def get_preprocessor():
    """Returns ColumnTransformer for numeric and categorical features."""
    num_features = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
    cat_features = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), num_features),
            ('cat', OneHotEncoder(sparse_output=False, handle_unknown='ignore'), cat_features)
        ],
        remainder='passthrough'
    )
    return preprocessor


def build_baseline_pipelines(preprocessor):
    """Constructs dictionary of sklearn Pipelines for 5 baseline algorithms."""
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Random Forest": RandomForestClassifier(random_state=42),
        "K-Nearest Neighbors": KNeighborsClassifier(),
        "Support Vector Machine": SVC(probability=True, random_state=42)
    }
    
    pipelines = {}
    for name, clf in models.items():
        pipelines[name] = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', clf)
        ])
        
    return pipelines


def benchmark_models(X_train, X_test, y_train, y_test, output_dir: Path):
    """Executes 5-fold CV and test set evaluation for all models."""
    preprocessor = get_preprocessor()
    pipelines = build_baseline_pipelines(preprocessor)
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scoring = ['accuracy', 'roc_auc']
    
    results_list = []
    confusion_matrices = {}
    roc_curves_data = {}
    
    print("=" * 80)
    print("PHASE 2A: LEAKAGE-FREE MULTI-MODEL BENCHMARK")
    print("=" * 80)
    print(f"Training Set Size : {X_train.shape[0]} samples")
    print(f"Test Set Size     : {X_test.shape[0]} samples")
    print("-" * 80)
    
    for name, pipeline in pipelines.items():
        # A. Stratified 5-Fold Cross-Validation on X_train / y_train
        cv_results = cross_validate(
            pipeline, X_train, y_train, cv=cv, scoring=scoring, return_train_score=False
        )
        
        cv_acc_mean = cv_results['test_accuracy'].mean()
        cv_acc_std = cv_results['test_accuracy'].std()
        cv_auc_mean = cv_results['test_roc_auc'].mean()
        cv_auc_std = cv_results['test_roc_auc'].std()
        
        # B. Fit pipeline on full training set
        pipeline.fit(X_train, y_train)
        
        # C. Evaluate once on untouched test set
        y_pred = pipeline.predict(X_test)
        y_proba = pipeline.predict_proba(X_test)[:, 1]
        
        test_acc = accuracy_score(y_test, y_pred)
        test_prec = precision_score(y_test, y_pred, zero_division=0)
        test_rec = recall_score(y_test, y_pred, zero_division=0)
        test_f1 = f1_score(y_test, y_pred, zero_division=0)
        test_auc = roc_auc_score(y_test, y_proba)
        
        # D. Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        confusion_matrices[name] = cm
        
        # E. ROC Curve data
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_curves_data[name] = (fpr, tpr, test_auc)
        
        results_list.append({
            "Model": name,
            "CV Accuracy": f"{cv_acc_mean:.4f} ± {cv_acc_std:.4f}",
            "CV ROC-AUC": f"{cv_auc_mean:.4f} ± {cv_auc_std:.4f}",
            "Test Accuracy": test_acc,
            "Test Precision": test_prec,
            "Test Recall": test_rec,
            "Test F1-Score": test_f1,
            "Test ROC-AUC": test_auc,
            "_cv_auc_raw": cv_auc_mean  # Used for sorting
        })
        
    # F. Create pandas DataFrame & Sort by CV ROC-AUC -> Test ROC-AUC -> Test F1-Score
    df_results = pd.DataFrame(results_list)
    df_results = df_results.sort_values(
        by=["_cv_auc_raw", "Test ROC-AUC", "Test F1-Score"],
        ascending=[False, False, False]
    ).reset_index(drop=True)
    
    # Remove helper column used for sorting
    display_df = df_results.drop(columns=["_cv_auc_raw"])
    
    print("\nBENCHMARK RESULTS TABLE (Sorted by CV ROC-AUC -> Test ROC-AUC -> Test F1-Score):")
    print("=" * 100)
    print(display_df.to_string(index=False))
    print("=" * 100)
    
    # Print Confusion Matrices
    print("\nCONFUSION MATRICES ON UNTOUCHED TEST SET (61 Patients):")
    print("-" * 80)
    for name in display_df["Model"]:
        cm = confusion_matrices[name]
        tn, fp, fn, tp = cm.ravel()
        print(f"[{name}]")
        print(f"  True Negatives (TN): {tn:2d} | False Positives (FP): {fp:2d}")
        print(f"  False Negatives (FN): {fn:2d} | True Positives  (TP): {tp:2d}")
        print(f"  Confusion Matrix Grid:\n{cm}\n")
        
    # Plot ROC Comparison
    plt.figure(figsize=(9, 7))
    for name in display_df["Model"]:
        fpr, tpr, auc_val = roc_curves_data[name]
        plt.plot(fpr, tpr, lw=2, label=f"{name} (AUC = {auc_val:.3f})")
        
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Chance (AUC = 0.500)')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (1 - Specificity)', fontsize=12)
    plt.ylabel('True Positive Rate (Sensitivity / Recall)', fontsize=12)
    plt.title('Phase 2A: ROC Curve Comparison across Baseline Classification Models', fontsize=14, pad=15)
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    
    roc_plot_path = output_dir / "roc_comparison.png"
    plt.savefig(roc_plot_path, dpi=300)
    plt.close()
    print(f"ROC Curve Comparison Plot saved to: {roc_plot_path}")
    
    return display_df, confusion_matrices, roc_curves_data


def main():
    base_dir = Path(__file__).resolve().parent.parent
    data_path = base_dir / "dataset" / "heart.csv"
    output_dir = base_dir / "models"
    output_dir.mkdir(exist_ok=True)
    
    X_train, X_test, y_train, y_test, df_clean = load_and_preprocess_data(data_path)
    benchmark_models(X_train, X_test, y_train, y_test, output_dir)


if __name__ == "__main__":
    main()
