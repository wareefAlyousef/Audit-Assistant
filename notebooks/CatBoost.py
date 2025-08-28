import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# 1️⃣ قراءة البيانات
df = pd.read_csv("cleanData100.csv")

# 2️⃣ تحديد الأعمدة
X = df.drop(["isFraud", "nameOrig", "nameDest"], axis=1)
y = df["isFraud"]

# 3️⃣ تحديد الأعمدة categorical
categorical_features = ["type"]  # CatBoost يتعرف عليها تلقائيًا

# 4️⃣ تقسيم البيانات
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 5️⃣ إنشاء Pool لـ CatBoost
train_pool = Pool(data=X_train, label=y_train, cat_features=categorical_features)
test_pool = Pool(data=X_test, label=y_test, cat_features=categorical_features)

# 6️⃣ إعداد النموذج
model = CatBoostClassifier(
    iterations=500,
    learning_rate=0.1,
    depth=6,
    eval_metric='AUC',
    random_seed=42,
    class_weights=[1, len(y_train[y_train==0]) / len(y_train[y_train==1])],  # يوازن الفئات
    verbose=50
)

# 7️⃣ تدريب النموذج
model.fit(train_pool, eval_set=test_pool, early_stopping_rounds=20)

# 8️⃣ التنبؤ
y_pred = model.predict(test_pool)

# 9️⃣ التقييم
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))

# 🔹 حفظ النموذج
model.save_model("catboost_fraud_model.cbm")
print("✅ النموذج تم حفظه في catboost_fraud_model.cbm")


import pandas as pd

# قراءة البيانات
df = pd.read_csv("cleanData100.csv")

# استخراج الصفوف اللي isFraud = 1
fraud_rows = df[df["isFraud"] == 1]

# عرضها
print(fraud_rows)

# لو تبغى تحفظها في ملف CSV منفصل
fraud_rows.to_csv("fraud_only.csv", index=False)
print("✅ تم حفظ الصفوف في fraud_only.csv")

import pandas as pd

# قراءة البيانات
df = pd.read_csv("cleanData100.csv")

# استخراج الصفوف اللي isFraud = 1
fraud_rows = df[df["isFraud"] == 1]

# حفظها في ملف Excel
fraud_rows.to_excel("fraud_only.xlsx", index=False)

print("✅ تم حفظ الصفوف الاحتيالية في ملف fraud_only.xlsx")
