from flask import Flask, render_template, request, jsonify, send_file
import pandas as pd
import os
import traceback
import numpy as np
from io import BytesIO
from datetime import datetime
from catboost import CatBoostClassifier

app = Flask(__name__)

# ---------------- Configuration ----------------
MODEL_PATH = "catboost_fraud_model.cbm"
UPLOAD_FOLDER = "uploads"
ALLOWED_EXTENSIONS = {'csv', 'xlsx'}
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# الأعمدة المطلوبة من المستخدم
USER_REQUIRED_COLUMNS = ["step", "type", "amount", "oldbalanceOrg", "newbalanceOrig"]

# ترتيب الأعمدة اللي استخدمناها أثناء التدريب
MODEL_FEATURE_ORDER = ["step", "type", "amount",  "oldbalanceOrg",
                       "newbalanceOrig",  "oldbalanceDest", "newbalanceDest"]

# ---------------- Helper Functions ----------------
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def detect_delimiter(file_path):
    with open(file_path, 'r', encoding='utf-8-sig') as f:
        first_line = f.readline()
    delimiters = [',',';','\t','|']
    delimiter_count = {d:first_line.count(d) for d in delimiters}
    return max(delimiter_count, key=delimiter_count.get)

def read_csv_with_auto_detect(file_path):
    delimiter = detect_delimiter(file_path)
    encodings = ['utf-8-sig','utf-8','latin-1','iso-8859-1','windows-1252']
    for enc in encodings:
        try:
            df = pd.read_csv(file_path, delimiter=delimiter, encoding=enc)
            if not df.empty:
                return df
        except: 
            continue
    return pd.read_csv(file_path, delimiter=delimiter)

def validate_dataframe(df):
    df.columns = df.columns.str.strip().str.replace('\ufeff','')
    missing_cols = [col for col in USER_REQUIRED_COLUMNS if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    if df.empty:
        raise ValueError("Uploaded file is empty")
    # تحويل الأعمدة الرقمية
    numeric_cols = ['amount','oldbalanceOrg','newbalanceOrig']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    return True

def prepare_for_prediction(df):
    df_prepared = df.copy()
    for col in MODEL_FEATURE_ORDER:
        if col not in df_prepared.columns:
            if col in ["nameOrig"]:
                df_prepared[col] = [f"C{1000000+i}" for i in range(len(df_prepared))]
            elif col in ["nameDest"]:
                df_prepared[col] = [f"C{2000000+i}" for i in range(len(df_prepared))]
            else:
                df_prepared[col] = 0.0
    df_prepared = df_prepared[MODEL_FEATURE_ORDER]
    return df_prepared

def encode_categoricals(df):
    df_encoded = df.copy()
    for col in ["type"]:
        df_encoded[col] = df_encoded[col].astype(str)
    return df_encoded

def predict_fraud(df, model_path=MODEL_PATH):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file {model_path} not found")
    model = CatBoostClassifier()
    model.load_model(model_path)
    preds = model.predict(df)
    probas = model.predict_proba(df)
    return preds, probas

def cleanup_file(file_path):
    if os.path.exists(file_path):
        os.remove(file_path)

def generate_analytics(df_result):
    analytics = {}
    analytics['total_count'] = len(df_result)
    analytics['fraud_count'] = int((df_result["predicted_fraud"]==1).sum())
    analytics['legit_count'] = analytics['total_count'] - analytics['fraud_count']
    analytics['fraud_rate'] = round((analytics['fraud_count']/analytics['total_count']*100),2) if analytics['total_count']>0 else 0
    if 'type' in df_result.columns:
        fraud_by_type = df_result[df_result['predicted_fraud']==1].groupby('type').size().to_dict()
        analytics['fraud_by_type'] = fraud_by_type
        analytics['most_common_fraud_type'] = max(fraud_by_type, key=fraud_by_type.get) if fraud_by_type else "None"
    if 'step' in df_result.columns:
        fraud_over_time = df_result[df_result['predicted_fraud']==1].groupby('step').size().to_dict()
        analytics['fraud_over_time'] = fraud_over_time
    if 'amount' in df_result.columns:
        fraud_amounts = df_result[df_result['predicted_fraud']==1]['amount']
        analytics['max_fraud_amount'] = round(fraud_amounts.max(),2) if not fraud_amounts.empty else 0
        analytics['avg_fraud_amount'] = round(fraud_amounts.mean(),2) if not fraud_amounts.empty else 0
        analytics['min_fraud_amount'] = round(fraud_amounts.min(),2) if not fraud_amounts.empty else 0
    return analytics

# ---------------- Routes ----------------
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/upload", methods=["POST"])
def upload():
    file_path = None
    try:
        if "file" not in request.files:
            return jsonify({"error":"No file uploaded"}), 400
        file = request.files["file"]
        if file.filename == "":
            return jsonify({"error":"No file selected"}), 400
        if not allowed_file(file.filename):
            return jsonify({"error":f"Unsupported file type. Allowed: {', '.join(ALLOWED_EXTENSIONS)}"}), 400
        file_path = os.path.join(UPLOAD_FOLDER,file.filename)
        file.save(file_path)
        if file.filename.endswith(".csv"):
            df_original = read_csv_with_auto_detect(file_path)
        else:
            df_original = pd.read_excel(file_path)
        validate_dataframe(df_original)
        df_for_model = prepare_for_prediction(df_original)
        df_encoded = encode_categoricals(df_for_model)
        preds, probas = predict_fraud(df_encoded)
        df_result = df_original.copy()
        df_result["predicted_fraud"] = preds
        df_result["fraud_probability"] = [round(p[1]*100,2) for p in probas]
        analytics = generate_analytics(df_result)
        expected_cols = USER_REQUIRED_COLUMNS + ["predicted_fraud","fraud_probability"]
        available_cols = [c for c in expected_cols if c in df_result.columns]
        data = df_result[available_cols].head(100).replace({np.nan: None}).to_dict(orient="records")
        return jsonify({
            "success": True,
            "data": data,
            "analytics": analytics,
            "message": f"Processed {analytics['total_count']} transactions. Found {analytics['fraud_count']} potential fraud cases ({analytics['fraud_rate']}%)."
        })
    except Exception as e:
        app.logger.error(traceback.format_exc())
        return jsonify({"error":str(e)}), 500
    finally:
        if file_path:
            cleanup_file(file_path)

@app.route("/download", methods=["POST"])
def download():
    try:
        data = request.get_json()
        if not data or 'results' not in data:
            return jsonify({"error":"No data provided"}), 400
        df = pd.DataFrame(data['results'])
        output = BytesIO()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"fraud_detection_results_{timestamp}.xlsx"
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            df.to_excel(writer, sheet_name='Results', index=False)
            summary_data = {
                'Metric': ['Total Transactions','Fraudulent Transactions','Legitimate Transactions','Fraud Rate'],
                'Value': [
                    data.get('total_count',0),
                    data.get('fraud_count',0),
                    data.get('total_count',0)-data.get('fraud_count',0),
                    f"{data.get('fraud_rate',0)}%"
                ]
            }
            pd.DataFrame(summary_data).to_excel(writer,sheet_name='Summary',index=False)
        output.seek(0)
        return send_file(output, as_attachment=True, download_name=filename,
                         mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')
    except Exception as e:
        app.logger.error(traceback.format_exc())
        return jsonify({"error":str(e)}), 500

@app.route("/health")
def health_check():
    return jsonify({
        "status":"healthy",
        "model_loaded": os.path.exists(MODEL_PATH),
        "model_path": MODEL_PATH
    })

@app.route("/model/features")
def model_features():
    if not os.path.exists(MODEL_PATH):
        return jsonify({"error":"Model file not found"}), 404
    try:
        model = CatBoostClassifier()
        model.load_model(MODEL_PATH)
        return jsonify({
            "features": MODEL_FEATURE_ORDER,
            "feature_count": len(MODEL_FEATURE_ORDER),
            "feature_order": "Exact order expected by the model"
        })
    except Exception as e:
        return jsonify({"error":str(e)}), 500

# ---------------- Run App ----------------
if __name__ == "__main__":
    if not os.path.exists(MODEL_PATH):
        print(f"Warning: Model file '{MODEL_PATH}' not found. Prediction will fail.")
    app.run(debug=True, host='0.0.0.0', port=5000)
