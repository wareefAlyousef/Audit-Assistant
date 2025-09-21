# -*- coding: utf-8 -*-
"""
fraud_model_vscode_ready.py
VS Code ready version using your Excel dataset
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, precision_recall_curve, average_precision_score, precision_score, recall_score, f1_score, roc_curve, auc
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import seaborn as sns

# -----------------------------
# 1️⃣ Load Excel dataset
# -----------------------------
file_path = r"C:\Users\lenovo\OneDrive\المستندات\GitHub\Audit-Assistant\notebooks\dataset.xlsx"
if not os.path.exists(file_path):
    raise FileNotFoundError(f"File not found: {file_path}")

df = pd.read_excel(file_path, engine='openpyxl')
print("✅ Excel file loaded successfully!")

# -----------------------------
# 2️⃣ Target and features
# -----------------------------
TARGET = 'is_suspicious'

# Use only numeric columns to avoid errors in IsolationForest/LOF
numeric_cols = ["amount", "old_balance", "new_balance"]
X = df[numeric_cols]
y = df[TARGET]

# Encode target if object
if y.dtypes == "object":
    y = LabelEncoder().fit_transform(y)

# -----------------------------
# 3️⃣ Train-test split
# -----------------------------
RANDOM_STATE = 42
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

# -----------------------------
# 4️⃣ Evaluation functions
# -----------------------------
def evaluate_binary(y_true, y_scores, threshold=0.5):
    res = {}
    res['roc_auc'] = roc_auc_score(y_true, y_scores)
    res['avg_precision'] = average_precision_score(y_true, y_scores)
    y_pred = (y_scores >= threshold).astype(int)
    res['precision'] = precision_score(y_true, y_pred)
    res['recall'] = recall_score(y_true, y_pred)
    res['f1'] = f1_score(y_true, y_pred)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    res['confusion'] = {'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp)}
    return res

def plot_confusion(y_true, y_scores, model_name, threshold=0.5):
    y_pred = (y_scores >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Normal","Fraud"], yticklabels=["Normal","Fraud"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"Confusion Matrix — {model_name}")
    plt.show()

def plot_precision_recall(y_true, y_scores, model_name):
    precision, recall, _ = precision_recall_curve(y_true, y_scores)
    ap = average_precision_score(y_true, y_scores)
    plt.figure(figsize=(6,4))
    plt.plot(recall, precision)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"Precision-Recall — {model_name} (AP={ap:.3f})")
    plt.grid(True)
    plt.show()

def plot_roc_curve(y_true, y_scores, model_name):
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(6,4))
    plt.plot(fpr, tpr, label=f'ROC curve (AUC={roc_auc:.3f})')
    plt.plot([0,1],[0,1],'--', color='gray')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve — {model_name}")
    plt.legend()
    plt.grid(True)
    plt.show()

# -----------------------------
# 5️⃣ Isolation Forest
# -----------------------------
iso = IsolationForest(n_estimators=200, random_state=RANDOM_STATE)
iso.fit(X_train_scaled)
iso_scores = -iso.decision_function(X_test_scaled)
res_iso = evaluate_binary(y_test.values, iso_scores)
plot_confusion(y_test.values, iso_scores, "IsolationForest")
plot_precision_recall(y_test.values, iso_scores, "IsolationForest")
plot_roc_curve(y_test.values, iso_scores, "IsolationForest")

# -----------------------------
# 6️⃣ Local Outlier Factor
# -----------------------------
lof = LocalOutlierFactor(n_neighbors=20, novelty=True)
lof.fit(X_train_scaled)
lof_scores = -lof.decision_function(X_test_scaled)
res_lof = evaluate_binary(y_test.values, lof_scores)
plot_confusion(y_test.values, lof_scores, "LOF")
plot_precision_recall(y_test.values, lof_scores, "LOF")
plot_roc_curve(y_test.values, lof_scores, "LOF")

# -----------------------------
# 7️⃣ XGBoost + SMOTE
# -----------------------------
sm = SMOTE(random_state=RANDOM_STATE)
X_res, y_res = sm.fit_resample(X_train_scaled, y_train)

xgb_sm = XGBClassifier(n_estimators=200, learning_rate=0.05, max_depth=6,
                       use_label_encoder=False, eval_metric='logloss', random_state=RANDOM_STATE)
xgb_sm.fit(X_res, y_res)
xgb_sm_scores = xgb_sm.predict_proba(X_test_scaled)[:,1]
res_xgb_sm = evaluate_binary(y_test.values, xgb_sm_scores)
plot_confusion(y_test.values, xgb_sm_scores, "XGBoost + SMOTE")
plot_precision_recall(y_test.values, xgb_sm_scores, "XGBoost + SMOTE")
plot_roc_curve(y_test.values, xgb_sm_scores, "XGBoost + SMOTE")

# -----------------------------
# 8️⃣ Summary Table
# -----------------------------
results = pd.DataFrame([
    {'model':'IsolationForest', **res_iso},
    {'model':'LOF', **res_lof},
    {'model':'XGBoost + SMOTE', **res_xgb_sm},
])

print("\n✅ Model Performance Summary:")
print(results[['model','roc_auc','avg_precision','precision','recall','f1']])
