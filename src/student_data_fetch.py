import numpy as np
import pandas as pd


def student_data_generated(n):
    # Generate Features
    assignment_score = np.random.randint(50, 101, n)
    attendance = np.random.randint(60, 101, n)
    quiz_score = np.random.randint(40, 101, n)
    midterm_score = np.random.randint(40, 101, n)
    study_hours = np.random.randint(1, 11, n)

    # Generate Target Variable
    final_grade = (
        assignment_score * 0.30 +
        attendance * 0.15 +
        quiz_score * 0.20 +
        midterm_score * 0.25 +
        study_hours * 1.0
    )

    # Add Random Noise
    final_grade += np.random.normal(
        loc=0,
        scale=3,
        size=n
    )

    # Keep Grades Between 0 and 100
    final_grade = np.clip(
        final_grade,
        0,
        100
    )

    # Create DataFrame
    df = pd.DataFrame({
        "assignment_score": assignment_score,
        "attendance": attendance,
        "quiz_score": quiz_score,
        "midterm_score": midterm_score,
        "study_hours": study_hours,
        "final_grade": final_grade.round(2)
    })

    return df


def main():
    n = 2000

    df = student_data_generated(n)

    print("\nFirst 5 Rows:")
    print(df.head())

    print("\nDataset Shape:")
    print(df.shape)

    df.to_csv(
        r"data/student_data.csv",
        index=False
    )

    print("\nDataset Saved Successfully!")


if __name__ == "__main__":
    main()