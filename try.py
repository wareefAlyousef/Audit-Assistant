import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder

# مسار الملف
file_path = "transactions_data100.csv"  # غيّريه حسب الملف عندك

# قراءة الملف مباشرة
df = pd.read_csv(file_path)
print("✅ الملف قُرأ بنجاح. الأعمدة:")
print(df.columns.tolist())

# اختيار الأعمدة اللي يحتاجها النموذج
feature_cols = ['step', 'type', 'amount', 'oldbalanceOrg', 'newbalanceOrig',
                'oldbalanceDest', 'newbalanceDest']

# فصل الـ features والـ target
X = df[feature_cols]
y = df['isFraud']  # ممكن تستخدمها لو حاب تتأكد من صحة النموذج

# تحويل الأعمدة الفئوية (categorical) إلى أرقام
X_encoded = X.copy()
le = LabelEncoder()
X_encoded['type'] = le.fit_transform(X_encoded['type'])

# تحميل نموذج CatBoost
model = joblib.load("catboost_model.joblib")  # تأكدي اسم الملف عندك

# التنبؤ
preds = model.predict(X_encoded)

# إضافة التنبؤات للـ DataFrame
df['predicted_fraud'] = preds

# تصفية التنبؤات الاحتيالية فقط
fraud_df = df[df['predicted_fraud'] == 1]

# طباعة النتائج
print("\n✅ كل التنبؤات:")
print(df.head())

print("\n✅ التنبؤات الاحتيالية فقط:")
print(fraud_df.head())

# حفظ النتائج في ملف Excel إذا حبيت
df.to_excel("predictions.xlsx", index=False)
fraud_df.to_excel("fraud_only.xlsx", index=False)
print("\n✅ تم حفظ النتائج في predictions.xlsx و fraud_only.xlsx")
