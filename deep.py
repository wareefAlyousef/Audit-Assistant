import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import os
import matplotlib.pyplot as plt
import seaborn as sns

# المسار
file_path = "notebooks\\Transactions Data.csv"  # تأكد من المسار

# تحقق من وجود الملف قبل أي عملية
if os.path.exists(file_path):
    df = pd.read_csv(file_path)
    print("File loaded successfully!")

    # عرض أسماء الأعمدة وأول 5 صفوف
    print("\nColumns:", df.columns)
    print(df.head())

    # توزيع العمود الهدف
    plt.figure(figsize=(6,4))
    sns.countplot(x="isFraud", data=df)
    plt.title("Distribution of isFraud")
    plt.show()

    # افترض أن العمود الهدف اسمه isFraud
    X = df.drop("isFraud", axis=1)
    y = df["isFraud"]

    # معالجة القيم النصية
    for col in X.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))

    # التطبيع
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # تقسيم البيانات
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # بناء المودل
    model = Sequential([
        Dense(128, activation="relu", input_shape=(X_train.shape[1],)),
        Dropout(0.3),
        Dense(64, activation="relu"),
        Dropout(0.3),
        Dense(1, activation="sigmoid")
    ])

    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

    # التدريب
    history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2)

    # التقييم
    loss, acc = model.evaluate(X_test, y_test)
    print(f"Test Accuracy: {acc:.2f}")

    # --- Visualization of Training ---
    plt.figure(figsize=(12,5))

    # Loss
    plt.subplot(1,2,1)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title("Model Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()

    # Accuracy
    plt.subplot(1,2,2)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title("Model Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()

    plt.show()

    # Correlation Heatmap
    plt.figure(figsize=(10,8))
    sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap="coolwarm")
    plt.title("Correlation Matrix")
    plt.show()

else:
    print("File not found:", file_path)

