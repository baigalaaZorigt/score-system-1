from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load model bundle
bundle = joblib.load("loan_scoring_model.pkl")

model = bundle["model"]
scaler = bundle["scaler"]

# ⚠️ LabelEncoder-ийг ашиглахгүй → safe mapping ашиглана
employment_map = {
    "Хувийн салбар": 0,
    "Төрийн алба": 1,
    "Хувиараа": 2
}


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    # INPUT
    income = float(request.form["salary"])
    employment_years = float(request.form["experience"])
    amount = float(request.form["amount"])
    employment_type = request.form["employment"]

    # SAFE ENCODING
    employment_encoded = employment_map.get(employment_type, 0)

    # FEATURE ENGINEERING
    ratio = amount / income if income != 0 else 0
    annual_dti = (amount / (income * 12)) * 100 if income != 0 else 0
    log_income = np.log1p(income)
    log_amount = np.log1p(amount)

    # FINAL FEATURES (match training order!)
    features = np.array([[
        income,
        employment_years,
        amount,
        ratio,
        annual_dti,
        log_income,
        log_amount,
        employment_encoded
    ]])

    # SCALE
    features = scaler.transform(features)

    # PREDICT
    score = model.predict(features)[0]
    score = int(max(0, min(1000, round(score))))

    # DECISION
    if score >= 700:
        decision = "ЗӨВШӨӨРӨХ"
        color = "#16a34a"

    elif score >= 450:
        decision = "ГАР ШАЛГАЛТ"
        color = "#f59e0b"

    else:
        decision = "ТАТГАЛЗАХ"
        color = "#dc2626"

    return render_template(
        "index.html",
        score=score,
        decision=decision,
        color=color
    )


if __name__ == "__main__":
    app.run(debug=True)