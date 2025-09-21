import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# تحميل البيانات
file_path = "notebooks\Transactions Data.csv"
df = pd.read_csv(file_path)

# نتأكد إن عندنا العمود isFraud
print(df['isFraud'].value_counts())

# مقارنة المبالغ بين الاحتيالي والعادي
plt.figure(figsize=(8,5))
sns.boxplot(x="isFraud", y="amount", data=df)
plt.title("Distribution of Amount in Fraud vs Non-Fraud")
plt.show()

# مقارنة أنواع العمليات
plt.figure(figsize=(8,5))
sns.countplot(x="type", hue="isFraud", data=df)
plt.title("Transaction Type vs Fraud")
plt.show()

# متوسط الرصيد قبل العملية
print("\n📌 Average balances comparison:")
print(df.groupby("isFraud")[["oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest"]].mean())

# إحصائيات عامة للمبالغ
print("\n📌 Amount stats:")
print(df.groupby("isFraud")["amount"].describe())
