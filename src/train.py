import pandas as pd
import os
import pickle
import warnings

import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

from xgboost import XGBClassifier

# =========================
# CLEAN OUTPUT
# =========================
warnings.filterwarnings("ignore")

# =========================
# PATH SETUP
# =========================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

data_path = os.path.join(BASE_DIR, "data", "intern_performance_final_data.csv")
model_dir = os.path.join(BASE_DIR, "model")
plots_dir = os.path.join(BASE_DIR, "plots")

os.makedirs(model_dir, exist_ok=True)
os.makedirs(plots_dir, exist_ok=True)

# =========================
# LOAD DATA
# =========================
df = pd.read_csv(data_path)

print(df["outcome"].value_counts())
print(df["outcome"].value_counts(normalize=True))

# =========================
# ENCODE TARGET
# =========================
df["outcome"] = df["outcome"].map({
    "Struggle": 0,
    "Excel": 1
})

# =========================
# FEATURES
# =========================
features = [
    "task_completion_time",
    "feedback_rating",
    "attendance_rate"
]

X = df[features]
y = df["outcome"]

# =========================
# TRAIN-TEST SPLIT
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# =========================
# RANDOM FOREST MODEL
# =========================
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

rf_pred = rf.predict(X_test)
rf_cm = confusion_matrix(y_test, rf_pred)

print("\n🌲 Random Forest Results")
print("Accuracy:", accuracy_score(y_test, rf_pred))
print("Confusion Matrix:\n", rf_cm)

# =========================
# XGBOOST MODEL
# =========================
xgb = XGBClassifier(
    n_estimators=150,
    learning_rate=0.1,
    max_depth=4,
    random_state=42,
    eval_metric='logloss'
)

xgb.fit(X_train, y_train)

xgb_pred = xgb.predict(X_test)
xgb_cm = confusion_matrix(y_test, xgb_pred)

print("\n🚀 XGBoost Results")
print("Accuracy:", accuracy_score(y_test, xgb_pred))
print("Confusion Matrix:\n", xgb_cm)

# =========================
# SAVE MODELS
# =========================
with open(os.path.join(model_dir, "random_forest_model.pkl"), "wb") as f:
    pickle.dump(rf, f)

with open(os.path.join(model_dir, "xgboost_model.pkl"), "wb") as f:
    pickle.dump(xgb, f)

with open(os.path.join(model_dir, "feature_names.pkl"), "wb") as f:
    pickle.dump(features, f)

print("\n✅ Models saved successfully in /model folder")

# =========================
# CONFUSION MATRIX PLOTS 
# =========================

# -------------------------
# Random Forest Plot
# -------------------------
plt.figure(figsize=(6, 4))
sns.heatmap(
    rf_cm,
    annot=True,
    fmt="d",
    xticklabels=["Struggle", "Excel"],
    yticklabels=["Struggle", "Excel"]
)

plt.title("Random Forest Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")

rf_path = os.path.join(plots_dir, "rf_confusion_matrix.png")
plt.savefig(rf_path, dpi=300, bbox_inches="tight")
plt.close()

print("Saved:", rf_path)

# -------------------------
# XGBoost Plot
# -------------------------
plt.figure(figsize=(6, 4))
sns.heatmap(
    xgb_cm,
    annot=True,
    fmt="d",
    xticklabels=["Struggle", "Excel"],
    yticklabels=["Struggle", "Excel"]
)

plt.title("XGBoost Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")

xgb_path = os.path.join(plots_dir, "xgb_confusion_matrix.png")
plt.savefig(xgb_path, dpi=300, bbox_inches="tight")
plt.close()

print("Saved:", xgb_path)