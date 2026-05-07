import pickle
import numpy as np
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model_path = os.path.join(BASE_DIR, "model", "random_forest_model.pkl")

with open(model_path, "rb") as f:
    model = pickle.load(f)

# Example input
sample = np.array([[5, 4.5, 90]])  
# [task_completion_time, feedback_rating, attendance_rate]

prediction = model.predict(sample)

result = "Excel" if prediction[0] == 1 else "Struggle"

print("Prediction:", result)