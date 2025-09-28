# -------------------------------
# Ensemble Fraud Detection - VS
# -------------------------------

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import StackingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, precision_recall_curve
import os

file_path = "notebooks\Transactions Data.csv"

if os.path.exists(file_path):
    df = pd.read_csv(file_path)
    print("File loaded successfully!")  


df = df.drop(columns=["nameOrig", "nameDest"], errors='ignore')

# تحويل الأعمدة النوعية إلى أرقام
le = LabelEncoder()
if 'type' in df.columns:
    df['type'] = le.fit_transform(df['type'])

# -------------------------------
# 3. تجهيز X, y
# -------------------------------
X = df.drop("isFraud", axis=1)
y = df["isFraud"]

# تقسيم البيانات
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# -------------------------------
# 4. بناء Ensemble
# -------------------------------
estimators = [
    ('xgb', xgb.XGBClassifier(eval_metric='logloss', n_jobs=-1)),
    ('rf', RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=42))
]

stack = StackingClassifier(
    estimators=estimators,
    final_estimator=LogisticRegression(max_iter=1000, class_weight="balanced"),
    n_jobs=-1
)

# تدريب
stack.fit(X_train, y_train)

# -------------------------------
# 5. التنبؤ بالاحتمالات
# -------------------------------
y_probs = stack.predict_proba(X_test)[:, 1]

# -------------------------------
# 6. اختيار أفضل threshold
# -------------------------------
precisions, recalls, thresholds = precision_recall_curve(y_test, y_probs)
f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-6)
best_idx = np.argmax(f1_scores)
best_threshold = thresholds[best_idx]

print(f"Threshold: {best_threshold:.4f}")
print(f"📈 F1 Score: {f1_scores[best_idx]:.4f}")

# -------------------------------
# 7. التنبؤ باستخدام أفضل threshold
# -------------------------------
y_pred_custom = (y_probs >= best_threshold).astype(int)

print("\n📊 Classification Report:\n", classification_report(y_test, y_pred_custom))
print("🔎 Confusion Matrix:\n", confusion_matrix(y_test, y_pred_custom))
print("🏆 ROC-AUC:", roc_auc_score(y_test, y_probs))
