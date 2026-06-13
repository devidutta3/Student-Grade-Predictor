import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)
import joblib


def load_dataset(path):
    return pd.read_csv(path)


def split_data(df):

    X = df[
        [
            "assignment_score",
            "attendance",
            "quiz_score",
            "midterm_score",
            "study_hours"
        ]
    ]

    y = df["final_grade"]

    return train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )


def train_model(X_train, y_train):

    model = LinearRegression()

    model.fit(
        X_train,
        y_train
    )

    return model


def evaluate_model(
    model,
    X_test,
    y_test
):

    predictions = model.predict(
        X_test
    )

    mae = mean_absolute_error(
        y_test,
        predictions
    )

    mse = mean_squared_error(
        y_test,
        predictions
    )

    r2 = r2_score(
        y_test,
        predictions
    )

    print(f"MAE : {mae:.2f}")
    print(f"MSE : {mse:.2f}")
    print(f"R² Score : {r2:.4f}")


def save_model(model):

    joblib.dump(
        model,
        "models/grade_model.pkl"
    )

    print(
        "\nModel Saved Successfully!"
    )


def main():

    df = load_dataset(
        "data/student_data.csv"
    )

    X_train, X_test, y_train, y_test = split_data(
        df
    )

    model = train_model(
        X_train,
        y_train
    )

    evaluate_model(
        model,
        X_test,
        y_test
    )

    save_model(model)


if __name__ == "__main__":
    main()