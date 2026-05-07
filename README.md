# 🎯 Intern Performance Prediction System

A machine learning project that predicts whether an intern will **Excel or Struggle** based on performance indicators such as:

- Task completion time  
- Feedback ratings  
- Attendance rate  

Built using **Python, Scikit-learn, Random Forest, XGBoost, and Streamlit**, this project demonstrates a complete end-to-end ML workflow:

👉 Data generation  
👉 Model training  
👉 Evaluation  
👉 Visualization  
👉 Deployment (Streamlit App)

---

## 📌 Project Overview

This system helps evaluate intern performance using machine learning instead of manual judgment.

### ✔ What it does:
- Predicts intern performance (Excel / Struggle)
- Trains Random Forest and XGBoost models
- Evaluates models using accuracy and confusion matrix
- Visualizes feature importance
- Provides Streamlit web interface

---

## ⚙️ Tech Stack

- Python  
- Pandas  
- NumPy  
- Scikit-learn  
- XGBoost  
- Matplotlib  
- Seaborn  
- Streamlit  
- Pickle  

---

## 📁 Project Structure

```text
ML_Project_1/
│
├── data/
│   └── intern_performance_final_data.csv
│
├── model/
│   ├── random_forest_model.pkl
│   ├── xgboost_model.pkl
│   └── feature_names.pkl
│
├── plots/
│   ├── rf_confusion_matrix.png
│   ├── rf_feature_importance.png
│   ├── xgb_confusion_matrix.png
│   └── xgb_feature_importance.png
│
├── src/
│   ├── predict.py
│   ├── train.py
│   └── feature_importance.py
│
├── app/
│   └── app.py
│
├── requirements.txt
├── generate_dataset.py
└── README.md

```

# 📊 Dataset Description

The dataset is synthetically generated.

### 🔹 Input Features:

- task_completion_time (hours)
- feedback_rating (1–5 scale)
- attendance_rate (%)

### 🔹 Target Variable:

| Label | Meaning |
|------|--------|
| Excel | High-performing intern |
| Struggle | Low-performing intern |

---

# ⚙️ Project Workflow

## 1️⃣ Data Generation
Synthetic dataset is created using mathematical rules and randomness.

## 2️⃣ Data Preprocessing
- Label encoding:
  - Excel → 1  
  - Struggle → 0  

## 3️⃣ Model Training
Two models are trained:

- 🌲 Random Forest Classifier  
- 🚀 XGBoost Classifier  

## 4️⃣ Model Evaluation
- Accuracy Score  
- Confusion Matrix  

## 5️⃣ Visualization
- Confusion matrix heatmaps  
- Feature importance plots  


## 6️⃣ Deployment
Streamlit web app for real-time predictions

---

# 🤖 Machine Learning Models

## 🌲 Random Forest
- Ensemble of decision trees  
- Reduces overfitting  
- Strong baseline model  
- Good for structured data  

## 🚀 XGBoost
- Gradient boosting model  
- More optimized and powerful  
- Captures complex relationships  
- Often slightly better than RF  

---

# 📊 Model Performance

| Model | Accuracy |
|------|--------|
| Random Forest | ~0.77 |
| XGBoost | ~0.76 |

---

# 📈 Visualizations

This project generates:

✔ Confusion Matrix (RF & XGB)  
✔ Feature Importance Graphs   

---

# 🚀 Installation Guide

## 1. Clone Repository
```bash
git clone https://github.com/MamoonaQuddus/Intern-Performance-Prediction-Model
cd ML_Project_1

```

### Step 2: Create Virtual Environment (Recommended)
```bash
python -m venv venv

# On Windows
venv\Scripts\activate

# On macOS/Linux
source venv/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Generate Dataset
```bash
python generate_dataset.py
```

**Output**: `intern_performance_final_data.csv` (500 records)

### Step 5: Train Models
```bash
python src/train.py
```

**Output**:
- `random_forest_model.pkl`
- `xgboost_model.pkl`
- `feature_names.pkl`

**Training Time**: ~1-2 minutes on a standard laptop

---

## 💻 Usage

### Launch Streamlit Dashboard
```bash
streamlit run app.py
```

The application will open in your default web browser at `http://localhost:8501`

## 🖥 Streamlit Features

- Input intern details  
- Predict Excel / Struggle  
- Compare RF vs XGBoost  
- View results instantly  

---

## 🔍 Key Insights

✔ Task completion time is the most important feature  
✔ Feedback rating strongly affects performance  
✔ Attendance has moderate impact  
✔ Both models perform similarly  

---

## 🚀 Future Improvements

- Add more features (GPA, skills, etc.)  
- Improve dataset realism  
- Deploy on Streamlit Cloud  
- Add SHAP explainability  
- Add batch CSV prediction  

---

## 🤝 Contributing

Contributions are welcome!

If you'd like to improve this project:
- Fork the repository
- Create a new branch
- Make your changes
- Submit a pull request

You can also open issues for bugs or suggestions.

---

## 📄 License

This project is licensed under the MIT License.

---

## 📞 Contact

For questions or suggestions, feel free to reach out:

- GitHub: https://github.com/MamoonaQuddus

---

## ⭐ Support

If you found this project helpful, consider giving it a ⭐ on GitHub!

---

**Built with ❤️ by Mamoona Quddus**
