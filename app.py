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

# ---------- Statistics ----------
def get_numeric_statistics(df):
    stats = {}
    numeric_cols = df.select_dtypes(include=['number']).columns
    for col in numeric_cols:
        stats[col] = {
            "mean": df[col].mean(),
            "median": df[col].median(),
            "std": df[col].std(),
            "min": df[col].min(),
            "max": df[col].max(),
            "missing": int(df[col].isna().sum())
        }
    return stats

def get_categorical_statistics(df):
    cat_stats = {}
    cat_cols = df.select_dtypes(exclude=['number']).columns
    for col in cat_cols:
        counts = df[col].value_counts()
        percentages = df[col].value_counts(normalize=True) * 100
        cat_stats[col] = [
            {"value": val, "count": int(counts[val]), "percentage": round(percentages[val], 2)}
            for val in counts.index
        ]
    return cat_stats

# ---------- Routes ----------
@app.route("/")
def index():
    return "Flask Statistics API is running ✅"

@app.route("/upload", methods=["POST"])
def upload():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    # Save file temporarily
    os.makedirs("uploads", exist_ok=True)
    file_path = os.path.join("uploads", file.filename)
    file.save(file_path)

    # Read file
    if file.filename.endswith(".csv"):
        df = pd.read_csv(file_path)
    elif file.filename.endswith(".xlsx"):
        df = pd.read_excel(file_path)
    else:
        return jsonify({"error": "Unsupported file type"}), 400

    # Generate statistics
    numeric_stats = get_numeric_statistics(df)
    categorical_stats = get_categorical_statistics(df)

    return jsonify({ 
        "numeric_statistics": numeric_stats,
        "categorical_statistics": categorical_stats
    })


if __name__ == "__main__":
    app.run(debug=True)

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