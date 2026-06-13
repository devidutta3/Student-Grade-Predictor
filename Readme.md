# 🎓 Student Grade Predictor API

A Machine Learning project that predicts a student's final grade based on academic performance metrics such as assignment scores, attendance, quiz scores, midterm scores, and study hours.

The project uses **Python**, **Pandas**, **NumPy**, **Scikit-Learn**, and **FastAPI** to build and deploy a regression model as a REST API.

---

## 🚀 Features

* Generate synthetic student performance datasets
* Perform Exploratory Data Analysis (EDA)
* Train a Machine Learning Regression Model
* Predict final student grades
* Automatic API documentation with Swagger UI
* Input validation using Pydantic
* FastAPI-powered REST API

---

## 🛠️ Tech Stack

* Python
* Pandas
* NumPy
* Scikit-Learn
* FastAPI
* Uvicorn
* Joblib

---

## 📂 Project Structure

```text
Student-Grade-Predictor/
│
├── dataset_generator.py
├── eda.py
├── train_model.py
├── app.py
├── student_data.csv
├── grade_model.pkl
├── requirements.txt
└── README.md
```

---

## 📊 Dataset Features

| Feature          | Description           |
| ---------------- | --------------------- |
| assignment_score | Assignment Marks      |
| attendance       | Attendance Percentage |
| quiz_score       | Quiz Marks            |
| midterm_score    | Midterm Exam Marks    |
| study_hours      | Weekly Study Hours    |
| final_grade      | Target Variable       |

---

## ⚙️ Installation

### Clone Repository

```bash
git clone https://github.com/your-username/student-grade-predictor.git

cd student-grade-predictor
```

### Create Virtual Environment

```bash
python -m venv venv
```

### Activate Environment

Windows:

```bash
venv\Scripts\activate
```

Linux/Mac:

```bash
source venv/bin/activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 📈 Generate Dataset

```bash
python dataset_generator.py
```

This creates:

```text
student_data.csv
```

---

## 🔍 Exploratory Data Analysis

```bash
python eda.py
```

Displays:

* Dataset Shape
* Dataset Information
* Summary Statistics

---

## 🤖 Train Model

```bash
python train_model.py
```

Outputs:

```text
MAE Score
MSE Score
R² Score
```

Generates:

```text
grade_model.pkl
```

---

## 🌐 Run API

```bash
uvicorn app:app --reload
```

Server:

```text
http://127.0.0.1:8000
```

Swagger Documentation:

```text
http://127.0.0.1:8000/docs
```

---

## 📝 Prediction Endpoint

### POST /predict

Request:

```json
{
  "assignment_score": 85,
  "attendance": 92,
  "quiz_score": 80,
  "midterm_score": 78,
  "study_hours": 5
}
```

Response:

```json
{
  "predicted_grade": 79.65,
  "grade_letter": "C"
}
```

---

## 🎯 Machine Learning Workflow

1. Data Generation
2. Data Analysis
3. Feature Selection
4. Train-Test Split
5. Model Training
6. Model Evaluation
7. Model Deployment using FastAPI

---

## 📌 Future Improvements

* Random Forest Regressor
* XGBoost Regressor
* Streamlit Frontend
* Docker Support
* Cloud Deployment
* Model Monitoring

---

## 👨‍💻 Author

Devidutta Das

Frontend Developer | Python Developer | Machine Learning Enthusiast

YouTube Channel: CodeUdaan
