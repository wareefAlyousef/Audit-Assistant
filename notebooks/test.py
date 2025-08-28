import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report

# 1. قراءة البيانات
df = pd.read_csv("transactions_data100.csv")   # عدّل الاسم إذا ملفك مختلف
print(df.shape)
print(df.info())

# 2. الأعمدة اللي نستخدمها
numeric_features = ["step", "amount", "oldbalanceOrg", "newbalanceOrig",
                    "oldbalanceDest", "newbalanceDest"]
categorical_features = ["type"]

# 3. تجهيز الأعمدة (ترميز + توحيد قيم)
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
    ]
)

# 4. بناء البايبلاين مع Isolation Forest
model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("isolation_forest", IsolationForest(
        n_estimators=200,
        contamination=0.02,   # نسبة الشذوذ المتوقعة (تقدر تعدلها)
        random_state=42,
        n_jobs=-1
    ))
])

# 5. تدريب النموذج
model.fit(df)

# 6. التنبؤ
preds = model.named_steps["isolation_forest"].predict(
    model.named_steps["preprocessor"].transform(df)
)

# Isolation Forest يرجع: -1 (شاذ) و 1 (طبيعي)
df["Anomaly"] = [1 if p == -1 else 0 for p in preds]

# 7. تقييم النموذج مقابل isFraud الحقيقي
print(classification_report(df["isFraud"], df["Anomaly"]))

# 8. حفظ النتائج
df.to_csv("transactions_with_anomalies.csv", index=False)
print("✅ النتائج انحفظت في ملف transactions_with_anomalies.csv")
