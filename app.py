from flask import Flask, request, jsonify, send_file
import pandas as pd
import numpy as np
import os
from io import BytesIO
from datetime import datetime
import joblib  # لحفظ وتحميل XGBoost أو أي scikit-learn
from catboost import CatBoostClassifier

app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ---------------- Helpers ----------------
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
