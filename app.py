
from flask import Flask, request, jsonify
import pandas as pd
import joblib
import os
import sqlite3
from datetime import datetime

app = Flask(__name__)

DATABASE_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "fraud_predictions.db"
)


def init_db():
    """Create the SQLite database and predictions table."""

    with sqlite3.connect(DATABASE_PATH) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                transaction_date TEXT NOT NULL,
                amount REAL NOT NULL,
                merchant_id TEXT NOT NULL,
                transaction_type TEXT NOT NULL,
                location TEXT NOT NULL,
                fraud_probability REAL NOT NULL,
                risk_level TEXT NOT NULL,
                prediction_timestamp TEXT NOT NULL
            )
        """)

        conn.commit()


def save_prediction(
    transaction_date,
    amount,
    merchant_id,
    transaction_type,
    location,
    fraud_probability,
    risk_level
):
    """Save transaction details and prediction to SQLite."""

    with sqlite3.connect(DATABASE_PATH) as conn:
        conn.execute("""
            INSERT INTO predictions (
                transaction_date,
                amount,
                merchant_id,
                transaction_type,
                location,
                fraud_probability,
                risk_level,
                prediction_timestamp
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            str(transaction_date),
            amount,
            str(merchant_id),
            transaction_type,
            location,
            fraud_probability,
            risk_level,
            datetime.now().isoformat(timespec="seconds")
        ))

        conn.commit()

init_db()

MODEL_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "models",
    "random_forest_model.joblib"
)

model = joblib.load(MODEL_PATH)

@app.route("/")
def home():
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Financial Fraud Detection</title>
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <style>
            body {
                font-family: Arial, sans-serif;
                background: #f1f5f9;
                margin: 0;
                padding: 30px 15px;
            }
            .container {
                max-width: 550px;
                margin: auto;
                background: white;
                padding: 25px;
                border-radius: 12px;
                box-shadow: 0 4px 15px #0001;
            }
            h1 { color: #17365d; text-align: center; }
            p { color: #475569; text-align: center; }
            label {
                display: block;
                margin-top: 15px;
                font-weight: bold;
            }
            input, select, button {
                box-sizing: border-box;
                width: 100%;
                padding: 11px;
                margin-top: 6px;
                border: 1px solid #cbd5e1;
                border-radius: 6px;
                font-size: 15px;
            }
            button {
                margin-top: 22px;
                background: #17365d;
                color: white;
                border: none;
                cursor: pointer;
            }
            button:hover { background: #24518a; }
            button:disabled { opacity: 0.6; cursor: wait; }
            #result {
                margin-top: 20px;
                padding: 15px;
                border-radius: 8px;
                background: #eff6ff;
                display: none;
            }
            .error { color: #b91c1c; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Financial Fraud Detection</h1>
            <p>Enter transaction details to estimate fraud risk.</p>

            <form id="fraudForm">
                <label>Transaction Date</label>
                <input type="date" id="date"
                       value="2024-10-15" required>

                <label>Transaction Amount</label>
                <input type="number" id="amount" value="2500"
                       min="0.01" step="0.01" required>

                <label>Merchant ID</label>
                <input type="number" id="merchant" value="101" required>

                <label>Transaction Type</label>
                <select id="type">
                    <option value="Purchase">Purchase</option>
                    <option value="Refund">Refund</option>
                </select>

                <label>Location</label>
                <select id="location">
                    <option>New York</option>
                    <option>Los Angeles</option>
                    <option>Chicago</option>
                    <option>Houston</option>
                    <option>San Diego</option>
                </select>

                <button type="submit" id="submitBtn">
                    Predict Fraud Risk
                </button>
            </form>

            <div id="result" aria-live="polite"></div>
        </div>

        <script>
            const form = document.getElementById("fraudForm");
            const result = document.getElementById("result");
            const button = document.getElementById("submitBtn");

            form.addEventListener("submit", async function(event) {
                event.preventDefault();

                button.disabled = true;
                button.textContent = "Predicting...";
                result.style.display = "block";
                result.textContent = "Processing transaction...";

                const transaction = {
                    TransactionDate: document.getElementById("date").value,
                    Amount: Number(document.getElementById("amount").value),
                    MerchantID: Number(document.getElementById("merchant").value),
                    TransactionType: document.getElementById("type").value,
                    Location: document.getElementById("location").value
                };

                try {
                    const response = await fetch("/predict", {
                        method: "POST",
                        headers: {"Content-Type": "application/json"},
                        body: JSON.stringify(transaction)
                    });

                    const data = await response.json();

                    if (!response.ok) {
                        throw new Error(data.error || "Prediction failed");
                    }

                    result.replaceChildren();

                    const heading = document.createElement("h3");
                    heading.textContent = "Prediction Result";
                    result.appendChild(heading);

                    const probability = document.createElement("p");
                    probability.textContent =
                        "Fraud Probability: " +
                        (data.fraud_probability * 100).toFixed(2) + "%";
                    result.appendChild(probability);

                    const risk = document.createElement("p");
                    risk.textContent = "Risk Level: " + data.risk_level;
                    result.appendChild(risk);

                    const message = document.createElement("p");
                    message.textContent = data.message;
                    result.appendChild(message);

                    const saved = document.createElement("p");
                    saved.textContent = data.database_status;
                    result.appendChild(saved);

                } catch (error) {
                    result.replaceChildren();

                    const errorText = document.createElement("p");
                    errorText.className = "error";
                    errorText.textContent = "Error: " + error.message;
                    result.appendChild(errorText);
                } finally {
                    button.disabled = false;
                    button.textContent = "Predict Fraud Risk";
                }
            });
        </script>
    </body>
    </html>
    """

@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "healthy",
        "message": "Fraud Detection API is running"
    })

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        if not data:
            return jsonify({
                "error": "Please provide transaction data"
            }), 400

        required_fields = [
            "Amount",
            "MerchantID",
            "TransactionType",
            "Location",
            "TransactionDate"
        ]

        missing_fields = [
            field for field in required_fields
            if field not in data
        ]

        if missing_fields:
            return jsonify({
                "error": "Missing required fields",
                "fields": missing_fields
            }), 400

        transaction_date = pd.to_datetime(
            data["TransactionDate"], errors="raise"
        )

        amount = float(data["Amount"])

        if not amount > 0:
            return jsonify({
                "error": "Amount must be greater than zero"
            }), 400

        merchant_id = data["MerchantID"]
        transaction_type = data["TransactionType"]
        location = data["Location"]

        transaction = pd.DataFrame([{
            "Amount": amount,
            "MerchantID": merchant_id,
            "TransactionType": transaction_type,
            "Location": location,
            "TransactionMonth": transaction_date.month,
            "TransactionDay": transaction_date.day,
            "TransactionDayOfWeek": transaction_date.dayofweek
        }])

        probability = float(
            model.predict_proba(transaction)[0, 1]
        )

        if probability <= 0.10:
            risk_level = "Low"
        elif probability <= 0.30:
            risk_level = "Medium"
        else:
            risk_level = "High"

        save_prediction(
            transaction_date=transaction_date,
            amount=amount,
            merchant_id=merchant_id,
            transaction_type=transaction_type,
            location=location,
            fraud_probability=probability,
            risk_level=risk_level
        )

        app.logger.info(
            "Prediction saved to SQLite: risk=%s, probability=%.4f",
            risk_level,
            probability
        )

        return jsonify({
            "fraud_probability": round(probability, 4),
            "risk_level": risk_level,
            "message": "Prediction generated successfully",
            "database_status": "Transaction and prediction saved successfully"
        })

    except (ValueError, TypeError):
        return jsonify({
            "error": "Invalid transaction data. Check the amount and date."
        }), 400

    except Exception:
        app.logger.exception("Prediction or database storage failed")
        return jsonify({
            "error": "Prediction or database storage failed. Check the server logs."
        }), 500



if __name__ == "__main__":
    app.run(debug=True, port=5000)
