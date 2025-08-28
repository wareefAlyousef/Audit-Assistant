import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import joblib

# 1️⃣ قراءة البيانات
df = pd.read_csv("cleanData100.csv")

# 2️⃣ اختيار الـ features والـ target
X = df.drop(["isFraud", "nameOrig", "nameDest"], axis=1)  # العمود type سيبقى
y = df["isFraud"]

# 3️⃣ تحويل العمود 'type' إلى category
X["type"] = X["type"].astype("category")

# 4️⃣ تقسيم البيانات
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 5️⃣ تحويل البيانات إلى DMatrix مع enable_categorical=True
dtrain = xgb.DMatrix(X_train, label=y_train, enable_categorical=True)
dtest = xgb.DMatrix(X_test, label=y_test, enable_categorical=True)

# 6️⃣ إعداد الباراميترات
params = {
    "objective": "binary:logistic",
    "eval_metric": "auc",
    "scale_pos_weight": len(y_train[y_train==0]) / len(y_train[y_train==1]),
    "max_depth": 6,
    "eta": 0.1,
    "subsample": 0.8,
    "colsample_bytree": 0.8
}

# 7️⃣ تدريب النموذج
model = xgb.train(
    params,
    dtrain,
    num_boost_round=300,
    evals=[(dtrain, "train"), (dtest, "test")],
    early_stopping_rounds=20,
    verbose_eval=50
)

# 8️⃣ التنبؤ
y_pred_prob = model.predict(dtest)
y_pred = [1 if p > 0.5 else 0 for p in y_pred_prob]

# 9️⃣ التقييم
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))

# 🔹 حفظ النموذج
joblib.dump(model, "xgboost_fraud_model.pkl")
print("✅ النموذج تم حفظه في xgboost_fraud_model.pkl")
