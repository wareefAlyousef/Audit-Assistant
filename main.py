import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder

# ---------- Functions ----------

def read_excel_csv(file_path):
    try:
        df = pd.read_csv(file_path)
        return df
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return None

def read_excel_xlsx(file_path):
    try:
        df = pd.read_excel(file_path)
        return df
    except Exception as e:
        print(f"Error reading Excel file: {e}")
        return None

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

# ---------- Main ----------

if __name__ == "__main__":
    file_path = "sample103_nolabel.csv"  # اسم ملفك الأصلي
    df_original = read_excel_csv(file_path)

    if df_original is not None:
        # نسخة معالجة فقط للتنبؤ
        df_normalized = normalize_iqr(df_original)
        df_encoded = encode_categoricals(df_normalized)

        # التنبؤ
        preds = predict_fraud(df_encoded)

        # رجع النتيجة للأصل
        df_result = df_original.copy()
        df_result["predicted_fraud"] = preds

        # حفظ مع تلوين الاحتيال بالأحمر
        def highlight_fraud(row):
            color = 'background-color: red' if row["predicted_fraud"] == 1 else ''
            return [color] * len(row)

        styled = df_result.style.apply(highlight_fraud, axis=1)
        styled.to_excel("predictions_highlighted.xlsx", index=False, engine="openpyxl")

        print("✅ تم إنشاء ملف predictions_highlighted.xlsx مع تلوين الصفوف الاحتيالية بالأحمر.")
