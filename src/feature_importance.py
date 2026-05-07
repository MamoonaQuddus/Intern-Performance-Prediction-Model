import pickle
import matplotlib.pyplot as plt
import os

# =========================
# PATHS
# =========================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

rf_path = os.path.join(BASE_DIR, "model", "random_forest_model.pkl")
xgb_path = os.path.join(BASE_DIR, "model", "xgboost_model.pkl")
feature_path = os.path.join(BASE_DIR, "model", "feature_names.pkl")

plots_dir = os.path.join(BASE_DIR, "plots")
os.makedirs(plots_dir, exist_ok=True)

# =========================
# LOAD MODELS
# =========================
with open(rf_path, "rb") as f:
    rf_model = pickle.load(f)

with open(xgb_path, "rb") as f:
    xgb_model = pickle.load(f)

with open(feature_path, "rb") as f:
    features = pickle.load(f)

# =========================
# IMPORTANCE VALUES
# =========================
rf_importance = rf_model.feature_importances_
xgb_importance = xgb_model.feature_importances_

# =========================
# 1️⃣ RANDOM FOREST PLOT
# =========================
plt.figure(figsize=(8, 5))
plt.barh(features, rf_importance)
plt.title("Random Forest Feature Importance")
plt.xlabel("Importance")

rf_path_img = os.path.join(plots_dir, "rf_feature_importance.png")
plt.savefig(rf_path_img, dpi=300, bbox_inches="tight")
plt.close()

print("Saved:", rf_path_img)

# =========================
# 2️⃣ XGBOOST PLOT
# =========================
plt.figure(figsize=(8, 5))
plt.barh(features, xgb_importance)
plt.title("XGBoost Feature Importance")
plt.xlabel("Importance")

xgb_path_img = os.path.join(plots_dir, "xgb_feature_importance.png")
plt.savefig(xgb_path_img, dpi=300, bbox_inches="tight")
plt.close()

print("Saved:", xgb_path_img)