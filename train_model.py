from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split


BASE_DIR = Path(__file__).resolve().parent
features = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal"]
data = pd.read_csv(BASE_DIR / "dataset" / "heart.csv")
X, y = data[features], data["target"]
model = RandomForestClassifier(n_estimators=500, class_weight="balanced", random_state=42, n_jobs=-1, min_samples_leaf=2)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(model, X, y, cv=cv, scoring="balanced_accuracy")
print(f"5-fold balanced accuracy: {scores.mean():.3f} ± {scores.std():.3f}")
X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
model.fit(X_train, y_train)
joblib.dump(model, BASE_DIR / "heart_disease_model.pkl")
print("Saved heart_disease_model.pkl")
