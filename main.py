from flask import Flask, request, jsonify, send_file
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
        if iqr == 0:
            df_normalized[col] = df[col] - median
        else:
            df_normalized[col] = (df[col] - median) / iqr
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

# ---------- Routes ----------

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    filename = file.filename

    try:
        # قراءة الملف
        if filename.endswith(".csv"):
            df_original = pd.read_csv(file)
        elif filename.endswith(".xlsx"):
            df_original = pd.read_excel(file)
        else:
            return jsonify({"error": "Unsupported file type"}), 400

        # نسخة معالجة
        df_normalized = normalize_iqr(df_original)
        df_encoded = encode_categoricals(df_normalized)

        # تنبؤ
        preds = predict_fraud(df_encoded)

        # رجع النتيجة للأصل
        df_result = df_original.copy()
        df_result["predicted_fraud"] = preds

        # طباعة النتائج في التيرمنال
        print("📊 كل البيانات:")
        print(df_result.head(20))   # يعرض أول 20 صف
        print("\n🚨 الصفوف الاحتيالية:")
        print(df_result[df_result["predicted_fraud"] == 1])

        # حفظ Excel مع تلوين الاحتيال
        def highlight_fraud(row):
            color = 'background-color: red' if row["predicted_fraud"] == 1 else ''
            return [color] * len(row)

        styled = df_result.style.apply(highlight_fraud, axis=1)
        output_path = "predictions_highlighted.xlsx"
        styled.to_excel(output_path, index=False, engine="openpyxl")

        return send_file(output_path, as_attachment=True)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)
