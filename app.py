from pathlib import Path

import joblib
import pandas as pd
from flask import Flask, jsonify, render_template, request


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "heart_disease_model.pkl"
FEATURES = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal",
]
RANGES = {
    "age": (1, 120), "sex": (0, 1), "cp": (0, 3), "trestbps": (50, 300),
    "chol": (50, 700), "fbs": (0, 1), "restecg": (0, 2), "thalach": (50, 250),
    "exang": (0, 1), "oldpeak": (0, 10), "slope": (0, 2), "ca": (0, 4), "thal": (0, 3),
}

app = Flask(__name__)
model = joblib.load(MODEL_PATH)


@app.get("/")
def index():
    return render_template("index.html")


@app.post("/predict")
def predict():
    payload = request.get_json(silent=True) or request.form.to_dict()
    values = {}
    errors = {}
    for feature in FEATURES:
        raw = payload.get(feature, "")
        try:
            value = float(raw)
            low, high = RANGES[feature]
            if not low <= value <= high:
                raise ValueError
            values[feature] = value
        except (TypeError, ValueError):
            errors[feature] = "Enter a valid value."
    if errors:
        return jsonify({"ok": False, "errors": errors}), 400

    row = pd.DataFrame([[values[name] for name in FEATURES]], columns=FEATURES)
    prediction = int(model.predict(row)[0])
    probability = float(model.predict_proba(row)[0][1]) if hasattr(model, "predict_proba") else None
    return jsonify({
        "ok": True,
        "prediction": prediction,
        "probability": round(probability * 100, 1) if probability is not None else None,
        "label": "Higher likelihood detected" if prediction else "Lower likelihood detected",
    })


if __name__ == "__main__":
    app.run(debug=True)
