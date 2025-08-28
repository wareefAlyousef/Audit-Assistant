from flask import Flask, render_template, request, send_file, jsonify
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder
import os

app = Flask(__name__)

# ---------- Functions ----------
def normalize_iqr(df):
    df_normalized = df.copy()
    numeric_cols = df.select_dtypes(include=['number']).columns

    for col in numeric_cols:
        median = df[col].median()
        iqr = df[col].quantile(0.75) - df[col].quantile(0.25)
        df_normalized[col] = (df[col] - median) / iqr if iqr != 0 else df[col] - median
    return df_normalized

def encode_categoricals(df):
    df_encoded = df.copy()
    cat_cols = df.select_dtypes(exclude="number").columns
    for col in cat_cols:
        le = LabelEncoder()
        df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
    return df_encoded

def predict_fraud(df, model_path="catboost_model.joblib"):
    model = joblib.load(model_path)
    preds = model.predict(df)
    return preds

# ----------  Rule-Based Checks ----------
def check_org(df):
    inconsistencies = []
    if "nameOrig" in df.columns:
        grouped = df.groupby("nameOrig")
        for name, group in grouped:
            if "oldbalanceOrg" in group.columns and "newbalanceOrig" in group.columns:
                balance_diff = group["newbalanceOrig"] - group["oldbalanceOrg"]
                if (balance_diff > 0).any():
                    inconsistencies.append({
                        "nameOrig": name,
                        "issue": "Balance increased after transaction",
                        "rows": group.index[balance_diff > 0].tolist()
                    })
    return inconsistencies

def check_dest(df):
    inconsistencies = []
    if "nameDest" in df.columns:
        grouped = df.groupby("nameDest")
        for name, group in grouped:
            if "oldbalanceDest" in group.columns and "newbalanceDest" in group.columns:
                balance_diff = group["newbalanceDest"] - group["oldbalanceDest"]
                if (balance_diff < 0).any():
                    inconsistencies.append({
                        "nameDest": name,
                        "issue": "Balance decreased after transaction (unexpected for destination)",
                        "rows": group.index[balance_diff < 0].tolist()
                    })
    return inconsistencies

def substractAmount(df):
    if "amount" in df.columns and "oldbalanceOrg" in df.columns and "newbalanceOrig" in df.columns:
        df["oldbalanceOrg"] -= df["amount"]
        df["newbalanceOrig"] -= df["amount"]
    return df

def substractAmountDest(df):
    if "amount" in df.columns and "oldbalanceDest" in df.columns and "newbalanceDest" in df.columns:
        df["oldbalanceDest"] -= df["amount"]
        df["newbalanceDest"] -= df["amount"]
    return df


def check_negative_balances(df):
    """Flag negative balances for origin or destination"""
    issues = []
    if "newbalanceOrig" in df.columns:
        negative_orig = df[df["newbalanceOrig"] < 0]
        for idx, row in negative_orig.iterrows():
            issues.append({
                "account": row["nameOrig"],
                "issue": "Origin balance negative",
                "row": idx
            })
    if "newbalanceDest" in df.columns:
        negative_dest = df[df["newbalanceDest"] < 0]
        for idx, row in negative_dest.iterrows():
            issues.append({
                "account": row["nameDest"],
                "issue": "Destination balance negative",
                "row": idx
            })
    return issues

def check_large_transactions(df, threshold=10000):
    """Flag transactions above a given threshold"""
    issues = []
    if "amount" in df.columns:
        large_tx = df[df["amount"] > threshold]
        for idx, row in large_tx.iterrows():
            issues.append({
                "account": row["nameOrig"],
                "issue": f"Transaction exceeds threshold {threshold}",
                "row": idx
            })
    return issues

def check_self_transfers(df):
    """Flag transactions where sender and recipient are the same"""
    issues = []
    if "nameOrig" in df.columns and "nameDest" in df.columns:
        self_tx = df[df["nameOrig"] == df["nameDest"]]
        for idx, row in self_tx.iterrows():
            issues.append({
                "account": row["nameOrig"],
                "issue": "Self-transfer detected",
                "row": idx
            })
    return issues

def check_zero_or_negative_amount(df):
    """Flag transactions with zero or negative amounts"""
    issues = []
    if "amount" in df.columns:
        invalid_tx = df[df["amount"] <= 0]
        for idx, row in invalid_tx.iterrows():
            issues.append({
                "account": row["nameOrig"],
                "issue": "Zero or negative transaction amount",
                "row": idx
            })
    return issues

def check_high_frequency(df, window=3):
    """
    Flag accounts that make multiple transactions within a short window (velocity check)
    window: number of consecutive steps to check
    """
    issues = []
    if "nameOrig" in df.columns and "step" in df.columns:
        grouped = df.groupby("nameOrig")
        for name, group in grouped:
            steps_sorted = group.sort_values("step")["step"].tolist()
            for i in range(len(steps_sorted) - window + 1):
                if steps_sorted[i + window - 1] - steps_sorted[i] < window:
                    indices = group.iloc[i:i+window].index.tolist()
                    issues.append({
                        "account": name,
                        "issue": f"High frequency of {window} transactions in short steps",
                        "rows": indices
                    })
                    break
    return issues

def Rule_Based_detect_fraud(df):
    # Initialize the fraud column
    df["is_fraud"] = 0

    # ---------- Helper function to increment fraud count ----------
    def mark_rows(rows):
        df.loc[rows, "is_fraud"] += 1

    # ---------- Apply Existing & New Rules ----------
    # 1. Origin balance inconsistency
    org_issues = check_org(df)
    for issue in org_issues:
        mark_rows(issue["rows"])

    # 2. Destination balance inconsistency
    dest_issues = check_dest(df)
    for issue in dest_issues:
        mark_rows(issue["rows"])

    # 3. Negative balances
    neg_bal_issues = check_negative_balances(df)
    for issue in neg_bal_issues:
        mark_rows([issue["row"]])

    # 4. Large transactions
    large_tx_issues = check_large_transactions(df)
    for issue in large_tx_issues:
        mark_rows([issue["row"]])

    # 5. Self-transfers
    self_tx_issues = check_self_transfers(df)
    for issue in self_tx_issues:
        mark_rows([issue["row"]])

    # 6. Zero or negative amounts
    zero_neg_amount_issues = check_zero_or_negative_amount(df)
    for issue in zero_neg_amount_issues:
        mark_rows([issue["row"]])

    # 7. High frequency / velocity checks
    high_freq_issues = check_high_frequency(df)
    for issue in high_freq_issues:
        mark_rows(issue["rows"])

    return df

# ---------- Routes ----------
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/upload", methods=["POST"])
def upload():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    # Save to temporary uploads folder
    os.makedirs("uploads", exist_ok=True)
    file_path = os.path.join("uploads", file.filename)
    file.save(file_path)

    # Read file safely
    if file.filename.endswith(".csv"):
        df_original = pd.read_csv(file_path)
    elif file.filename.endswith(".xlsx"):
        df_original = pd.read_excel(file_path)
    else:
        return jsonify({"error": "Unsupported file type"}), 400

    # Process
    df_normalized = normalize_iqr(df_original)
    df_encoded = encode_categoricals(df_normalized)
    preds = predict_fraud(df_encoded)

    df_result = df_original.copy()
    df_result["predicted_fraud"] = preds

    # Prepare JSON for frontend
    expected_cols = ["step", "type", "amount", "oldbalanceOrg", "newbalanceOrig", "predicted_fraud"]
    available_cols = [c for c in expected_cols if c in df_result.columns]

    data = df_result[available_cols].head(50).to_dict(orient="records")
    fraud_count = int((df_result["predicted_fraud"] == 1).sum())
    total_count = int(len(df_result))

    return jsonify({
        "data": data,
        "fraud_count": fraud_count,
        "total_count": total_count
    })

if __name__ == "__main__":
    app.run(debug=True)