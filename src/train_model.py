import pandas as pd
import matplotlib.pyplot as plt
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

    save_prediction_pie_chart(predictions)


def save_prediction_pie_chart(predictions, output_path="models/prediction_distribution_pie.png"):
    grade_labels = []
    for prediction in predictions:
        if prediction >= 90:
            grade_labels.append("A")
        elif prediction >= 80:
            grade_labels.append("B")
        elif prediction >= 70:
            grade_labels.append("C")
        elif prediction >= 60:
            grade_labels.append("D")
        else:
            grade_labels.append("F")

    counts = pd.Series(grade_labels).value_counts().reindex(["A", "B", "C", "D", "F"]).fillna(0)
    labels = counts.index.tolist()
    sizes = counts.values.tolist()
    colors = ["#4caf50", "#8bc34a", "#ffeb3b", "#ff9800", "#f44336"]

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.pie(
        sizes,
        labels=labels,
        autopct="%.1f%%",
        startangle=140,
        colors=colors,
        wedgeprops={"edgecolor": "white"}
    )
    ax.set_title("Predicted Grade Distribution")
    ax.axis("equal")

    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved prediction pie chart to: {output_path}")


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