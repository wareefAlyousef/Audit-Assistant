import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.utils import class_weight
from imblearn.over_sampling import SMOTE
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
from joblib import dump, load
import os

# -----------------------------
# تحميل البيانات
file_path = "Transactions Data.csv"
if not os.path.exists(file_path):
    raise FileNotFoundError("⚠️ الملف غير موجود، تأكد من المسار")
df = pd.read_csv(file_path)

# -----------------------------
# Feature Engineering
df['diffOrig'] = df['oldbalanceOrg'] - df['newbalanceOrig']
df['diffDest'] = df['oldbalanceDest'] - df['newbalanceDest']
df['amount_ratio'] = df['amount'] / (df['oldbalanceOrg'] + 1e-6)

# تحويل العمود type إلى أرقام
le_type = LabelEncoder()
df['type'] = le_type.fit_transform(df['type'])

# -----------------------------
# تجهيز البيانات
X = df.drop(['nameOrig', 'nameDest', 'isFraud'], axis=1)
y = df['isFraud']

# -----------------------------
# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# -----------------------------
# Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# -----------------------------
# التعامل مع عدم التوازن بواسطة SMOTE
print("قبل SMOTE:", np.bincount(y_train))
sm = SMOTE(random_state=42)
X_train_res, y_train_res = sm.fit_resample(X_train, y_train)
print("بعد SMOTE:", np.bincount(y_train_res))

# -----------------------------
# تدريب XGBoost
model = xgb.XGBClassifier(
    n_estimators=500,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    use_label_encoder=False,
    eval_metric='logloss',
    random_state=42
)

model.fit(X_train_res, y_train_res)

# -----------------------------
# التنبؤ
y_probs = model.predict_proba(X_test)[:,1]

# تحديد Threshold ديناميكي حسب ROC curve
from sklearn.metrics import roc_curve

fpr, tpr, thresholds = roc_curve(y_test, y_probs)
optimal_idx = np.argmax(tpr - fpr)
optimal_threshold = thresholds[optimal_idx]
print(f"🔎 Optimal Threshold: {optimal_threshold}")

y_pred = (y_probs >= optimal_threshold).astype(int)

# -----------------------------
# النتائج
print("\n📊 Classification Report:")
print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Not Fraud", "Fraud"],
            yticklabels=["Not Fraud", "Fraud"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - XGBoost")
plt.show()

# -----------------------------
# حفظ النموذج والـ Scaler والـ LabelEncoder والThreshold
dump(model, "xgboost_fraud_model.pkl")
dump(scaler, "scaler.pkl")
dump(le_type, "labelencoder_type.pkl")
np.save("best_threshold.npy", optimal_threshold)

print("✅ Model, Scaler, LabelEncoder, and Threshold saved successfully!")
