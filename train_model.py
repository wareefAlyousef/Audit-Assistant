import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

df = pd.read_csv("../data/transactions_data100.csv") 
X = df.drop("isFraud", axis=1)   # المتغيرات المستقلة
y = df["isFraud"]                # المتغير المستهدف (هل العملية احتيالية)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

joblib.dump(model, "fraud_model.pkl")

print("تم حفظ النموذج بنجاح في fraud_model.pkl")