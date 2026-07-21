from flask import Flask, render_template, request
import joblib

# Create Flask App
app = Flask(__name__)

# Load the trained model
model = joblib.load("heart_disease_model.pkl")


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    print("Predict function called")
    # Get input values from form
    features = [
        float(request.form["age"]),
        float(request.form["sex"]),
        float(request.form["cp"]),
        float(request.form["trestbps"]),
        float(request.form["chol"]),
        float(request.form["fbs"]),
        float(request.form["restecg"]),
        float(request.form["thalach"]),
        float(request.form["exang"]),
        float(request.form["oldpeak"]),
        float(request.form["slope"]),
        float(request.form["ca"]),
        float(request.form["thal"])
    ]

    # Debug prints
    print("=" * 60)
    print("Received Features:")
    print(features)

    prediction = model.predict([features])

    print("Prediction:", prediction)
    print("=" * 60)

    # Display result
    if prediction[0] == 1:
        result = "⚠ High Risk of Heart Disease"
    else:
        result = "✅ Low Risk of Heart Disease"

    return render_template("index.html", prediction=result)


if __name__ == "__main__":
    app.run(debug=True)