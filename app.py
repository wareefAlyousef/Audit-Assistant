<<<<<<< HEAD
from flask import Flask, request, jsonify, send_file
import pandas as pd
import numpy as np
import os
=======
from flask import Flask, render_template, request, jsonify, send_file
import pandas as pd
import os
import traceback
import numpy as np
>>>>>>> 585ec3ea849ea0c16a877b3cd94e1d40001176de
from io import BytesIO
from datetime import datetime
import joblib  # لحفظ وتحميل XGBoost أو أي scikit-learn
from catboost import CatBoostClassifier
import gspread
from google.oauth2.service_account import Credentials


app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

<<<<<<< HEAD
# ---------------- Helpers ----------------
=======
SCOPES = ["https://www.googleapis.com/auth/spreadsheets"]
creds = Credentials.from_service_account_file("anomalous-detect-9ab3bbadb4d4.json", scopes=SCOPES)
client = gspread.authorize(creds)

# افتح الشيت باستخدام ID من الرابط
SHEET_ID = "1rKCl2FrJSbQ6Ojcp4RJVb_Y5miGe7DivoQpkvGWvmZo"
sheet = client.open_by_key(SHEET_ID).sheet1


# الأعمدة المطلوبة من المستخدم
USER_REQUIRED_COLUMNS = ["step", "type", "amount", "oldbalanceOrg", "newbalanceOrig"]

# ترتيب الأعمدة اللي استخدمناها أثناء التدريب
MODEL_FEATURE_ORDER = ["step", "type", "amount",  "oldbalanceOrg",
                       "newbalanceOrig",  "oldbalanceDest", "newbalanceDest"]

# ---------------- Helper Functions ----------------
>>>>>>> 585ec3ea849ea0c16a877b3cd94e1d40001176de
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.',1)[1].lower() in ['csv','xlsx','pkl','cbm']

def read_file(file_path):
    if file_path.endswith('.csv'):
        return pd.read_csv(file_path)
    elif file_path.endswith('.xlsx'):
        return pd.read_excel(file_path)
    else:
        raise ValueError("Unsupported file type")

def prepare_dataframe(df, required_cols, feature_order):
    # إضافة أعمدة مفقودة
    for col in feature_order:
        if col not in df.columns:
            df[col] = 0.0
    df = df[feature_order]
    return df

def load_model(model_path):
    ext = model_path.split('.')[-1]
    if ext == 'pkl':
        return joblib.load(model_path)
    elif ext == 'cbm':
        model = CatBoostClassifier()
        model.load_model(model_path)
        return model
    else:
        raise ValueError("Unsupported model type")

def predict(df, model):
    ext = type(model).__name__
    if 'CatBoost' in ext:
        preds = model.predict(df)
        probas = model.predict_proba(df)[:,1]
    else:
        preds = model.predict(df)
        probas = model.predict_proba(df)[:,1]
    return preds, probas

<<<<<<< HEAD
=======
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

def write_fraud_to_sheet(df):
    # نحصر فقط الأعمدة المطلوبة
    required_cols = [
        "step", "type", "amount", "nameOrig", "oldbalanceOrg", "newbalanceOrig",
        "nameDest", "oldbalanceDest", "newbalanceDest", "predicted_fraud"
    ]
    fraud_df = df[required_cols]

    if fraud_df.empty:
        print("لا توجد بيانات")
        return

    # تحويل الداتا إلى ليست (للرفع)
    data = [fraud_df.columns.values.tolist()] + fraud_df.values.tolist()

    # مسح المحتوى السابق
    sheet.clear()

    # رفع البيانات
    sheet.update("A1", data)

>>>>>>> 585ec3ea849ea0c16a877b3cd94e1d40001176de
# ---------------- Routes ----------------
@app.route("/predict", methods=["POST"])
def predict_route():
    try:
        # رفع الملفات
        if 'model' not in request.files or 'data' not in request.files:
            return jsonify({"error":"Model and data files required"}),400

        model_file = request.files['model']
        data_file = request.files['data']

        model_path = os.path.join(UPLOAD_FOLDER, model_file.filename)
        data_path = os.path.join(UPLOAD_FOLDER, data_file.filename)
        model_file.save(model_path)
        data_file.save(data_path)

        # قراءة البيانات
        df = read_file(data_path)

        # تحميل النموذج
        model = load_model(model_path)

        # ترتيب الأعمدة حسب النموذج (مفترض تحفظ feature_order أثناء التدريب)
        if hasattr(model, 'feature_names_'):
            feature_order = model.feature_names_
        else:
<<<<<<< HEAD
            # لازم المستخدم يرسل feature_order إذا مش موجود
            feature_order = df.columns.tolist()
        df_prepared = prepare_dataframe(df, df.columns.tolist(), feature_order)

        # التنبؤ
        preds, probas = predict(df_prepared, model)
        df['predicted'] = preds
        df['probability'] = probas

        # تحليلات
        analytics = {
            'total': len(df),
            'fraud': int((df['predicted']==1).sum()),
            'legit': int((df['predicted']==0).sum())
        }

=======
            df_original = pd.read_excel(file_path)
        validate_dataframe(df_original)
        df_for_model = prepare_for_prediction(df_original)
        df_encoded = encode_categoricals(df_for_model)
        preds, probas = predict_fraud(df_encoded)
        df_result = df_original.copy()
        df_result["predicted_fraud"] = preds
        df_result["fraud_probability"] = [round(p[1]*100,2) for p in probas]
        write_fraud_to_sheet(df_result)
        analytics = generate_analytics(df_result)
        expected_cols = USER_REQUIRED_COLUMNS + ["predicted_fraud","fraud_probability"]
        available_cols = [c for c in expected_cols if c in df_result.columns]
        data = df_result[available_cols].head(100).replace({np.nan: None}).to_dict(orient="records")
>>>>>>> 585ec3ea849ea0c16a877b3cd94e1d40001176de
        return jsonify({
            "success": True,
            "analytics": analytics,
            "results": df.head(100).to_dict(orient='records')
        })
    finally:
        # تنظيف الملفات
        for f in [model_path,data_path]:
            if os.path.exists(f):
                os.remove(f)

if __name__=="__main__":
    app.run(debug=True, host='0.0.0.0', port=5000)
