# ======================================================
# Credit Card Fraud Detection System
# Hackathon Project – IILM University
# Team Lead: Md Asif
# ======================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

# ------------------------------------------------------
# 1️⃣ Create Synthetic Fraud Dataset (Imbalanced)
# ------------------------------------------------------

np.random.seed(42)

n_samples = 5000

data = {
    "TransactionAmount": np.random.normal(120, 60, n_samples),
    "TransactionTime": np.random.normal(50, 15, n_samples),
    "CustomerAge": np.random.randint(18, 70, n_samples),
    "LocationRiskScore": np.random.uniform(0, 1, n_samples),
    "Fraud": np.random.choice([0, 1], size=n_samples, p=[0.96, 0.04])  # 4% fraud
}

df = pd.DataFrame(data)

print("\nDataset Preview:\n")
print(df.head())

# ------------------------------------------------------
# 2️⃣ Check Class Distribution
# ------------------------------------------------------

print("\nClass Distribution:")
print(df["Fraud"].value_counts())

sns.countplot(x="Fraud", data=df)
plt.title("Fraud vs Non-Fraud Transactions")
plt.show()

# ------------------------------------------------------
# 3️⃣ Feature Selection
# ------------------------------------------------------

X = df.drop("Fraud", axis=1)
y = df["Fraud"]

# ------------------------------------------------------
# 4️⃣ Train-Test Split (Stratified)
# ------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ------------------------------------------------------
# 5️⃣ Feature Scaling
# ------------------------------------------------------

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ------------------------------------------------------
# 6️⃣ Logistic Regression Model
# ------------------------------------------------------

lr = LogisticRegression()
lr.fit(X_train, y_train)

y_pred_lr = lr.predict(X_test)

print("\n===== Logistic Regression Results =====")
print(classification_report(y_test, y_pred_lr))
print("ROC-AUC Score:", roc_auc_score(y_test, y_pred_lr))

# ------------------------------------------------------
# 7️⃣ Random Forest Model
# ------------------------------------------------------

rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

y_pred_rf = rf.predict(X_test)

print("\n===== Random Forest Results =====")
print(classification_report(y_test, y_pred_rf))
print("ROC-AUC Score:", roc_auc_score(y_test, y_pred_rf))

# ------------------------------------------------------
# 8️⃣ Confusion Matrix Visualization
# ------------------------------------------------------

cm = confusion_matrix(y_test, y_pred_rf)

plt.figure()
sns.heatmap(cm, annot=True, fmt="d")
plt.title("Confusion Matrix - Random Forest")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# ------------------------------------------------------
# 9️⃣ Sample Fraud Prediction
# ------------------------------------------------------

sample_transaction = np.array([[200, 45, 30, 0.8]])  # Example input
sample_transaction = scaler.transform(sample_transaction)

prediction = rf.predict(sample_transaction)

print("\nSample Transaction Prediction:")
if prediction[0] == 1:
    print("⚠️ Fraudulent Transaction Detected")
else:
    print("✅ Legitimate Transaction")
