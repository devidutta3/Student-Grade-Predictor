from fastapi import FastAPI
from pydantic import BaseModel, Field
import pandas as pd
import joblib


# Load Trained Model
model = joblib.load("models/grade_model.pkl")


# Create FastAPI App
app = FastAPI(
    title="Student Grade Predictor API",
    description="Predict student final grades using ML",
    version="1.0.0"
)


# Input Schema
class StudentData(BaseModel):
    assignment_score: float = Field(..., ge=0, le=100)
    attendance: float = Field(..., ge=0, le=100)
    quiz_score: float = Field(..., ge=0, le=100)
    midterm_score: float = Field(..., ge=0, le=100)
    study_hours: float = Field(..., ge=0, le=24)


@app.get("/")
def home():
    return {
        "message": "Welcome to Student Grade Predictor API"
    }


@app.post("/predict")
def predict(data: StudentData):

    input_data = pd.DataFrame(
        [[
            data.assignment_score,
            data.attendance,
            data.quiz_score,
            data.midterm_score,
            data.study_hours
        ]],
        columns=[
            "assignment_score",
            "attendance",
            "quiz_score",
            "midterm_score",
            "study_hours"
        ]
    )

    prediction = model.predict(input_data)[0]

    # Grade Classification
    if prediction >= 90:
        grade = "A"
    elif prediction >= 80:
        grade = "B"
    elif prediction >= 70:
        grade = "C"
    elif prediction >= 60:
        grade = "D"
    else:
        grade = "F"

    return {
        "predicted_grade": round(float(prediction), 2),
        "grade_letter": grade
    }