import pandas as pd
import numpy as np
import os

np.random.seed(42)

# ==============================
# DATA GENERATION FUNCTION
# ==============================
def generate_simple_data(n=500):
    data = []

    for i in range(n):

        # Hidden performance factor
        performance_base = np.random.normal(60, 15)

        # Task completion time (2–10 hours)
        task_time = max(2, min(10,
            10 - (performance_base / 20) + np.random.normal(0, 1)
        ))

        # Feedback rating (1–5)
        feedback = max(1, min(5,
            2 + (performance_base / 25) + np.random.normal(0, 0.5)
        ))

        # Attendance (70–100)
        attendance = max(70, min(100,
            75 + (performance_base / 3) + np.random.normal(0, 5)
        ))

        # Performance score
        performance_score = (
            (11 - task_time) * 10 * 0.4 +
            feedback * 20 * 0.35 +
            attendance * 0.25
        )

        # Noise
        performance_score += np.random.normal(0, 5)
        performance_score = round(max(45, min(100, performance_score)), 2)

        # Label
        outcome = "Excel" if performance_score >= 70 else "Struggle"

        data.append({
            "task_completion_time": round(task_time, 2),
            "feedback_rating": round(feedback, 2),
            "attendance_rate": round(attendance, 2),
            "performance_score": performance_score,
            "outcome": outcome
        })

    return pd.DataFrame(data)

# ==============================
# MAIN EXECUTION
# ==============================
if __name__ == "__main__":

    print("🚀 Generating intern performance dataset...")

    df = generate_simple_data(500)

    # ==============================
    # PATH SETUP 
    # ==============================
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(BASE_DIR, "data")

    os.makedirs(data_dir, exist_ok=True)

    file_path = os.path.join(data_dir, "intern_performance_final_data.csv")

    # ==============================
    # SAVE DATASET
    # ==============================
    df.to_csv(file_path, index=False)

    # ==============================
    # OUTPUT INFO
    # ==============================
    print(f"✓ Dataset created with {len(df)} records")
    print(f"✓ Saved to: {file_path}")

    print("\n" + "="*50)
    print("📊 DATASET SUMMARY")
    print("="*50)

    print("\nOutcome Distribution:")
    print(df["outcome"].value_counts())

    print("\nPerformance Score Stats:")
    print(df["performance_score"].describe())

    print("\n✅ Dataset generation complete!")