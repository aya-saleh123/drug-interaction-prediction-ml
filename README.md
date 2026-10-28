# Drug Interaction Prediction ML

A machine learning project for predicting potential drug interactions, associated side effects, and severity levels using trained ML models and a web-based interface.

## Overview

This project combines machine learning models with a Streamlit frontend and a FastAPI backend to provide predictions related to drug interactions and their potential effects.

### Features

- **Drug Interaction Prediction:** Predicts potential side effects associated with drug interactions.
- **Severity Prediction:** Estimates the severity level of predicted interactions.
- **Multiple Side-Effect Classification:** Uses trained classification models to predict possible side effects.
- **Drug Lookup:** Includes drug catalogs and lookup data to support the application.
- **Interactive Web Interface:** Streamlit-based frontend for interacting with the models.
- **REST API:** FastAPI backend for serving predictions and exposing model functionality.
- **Saved Models:** Loads trained models and preprocessing artifacts without requiring retraining.

## Tech Stack

- **Language:** Python
- **Machine Learning:** scikit-learn
- **Frontend:** Streamlit
- **Backend:** FastAPI, Uvicorn
- **Data Processing:** Pandas, NumPy
- **Model Persistence:** Joblib
- **Text Processing:** TF-IDF

## Project Structure

```text
drug-interaction-prediction-ml/
├── backend/
│   ├── artifacts/
│   │   ├── m3_logistic_regression.joblib
│   │   ├── severity_model.joblib
│   │   ├── structured_preprocessor.joblib
│   │   ├── tfidf_vectorizer.joblib
│   │   └── ...
│   ├── inference.py
│   ├── main.py
│   └── requirements.txt
├── frontend/
│   ├── app.py
│   └── requirements.txt
└── notebook/
    ├── El_Top_Wallahy_.ipynb
    └── Preprocessing&EDA.ipynb
```

## Installation and Setup

### 1. Clone the Repository

```bash
git clone https://github.com/aya-saleh123/drug-interaction-prediction-ml.git
cd drug-interaction-prediction-ml
```

### 2. Set Up the Backend

Create and activate a virtual environment:

```powershell
python -m venv backend/.venv
.\backend\.venv\Scripts\Activate.ps1
```

Install the backend dependencies:

```powershell
python -m pip install -r .\backend\requirements.txt
```

Start the API server from the project root:

```powershell
python -m uvicorn main:app --app-dir backend --reload
```

The backend will typically be available at `http://127.0.0.1:8000`.

API documentation: `http://127.0.0.1:8000/docs`

### 3. Set Up the Frontend

Open a **second terminal** in the project root and create a separate virtual environment:

```powershell
python -m venv frontend/.venv
.\frontend\.venv\Scripts\Activate.ps1
```

Install the frontend dependencies:

```powershell
python -m pip install -r .\frontend\requirements.txt
```

Launch Streamlit:

```powershell
python -m streamlit run .\frontend\app.py
```

The frontend will typically be available at `http://localhost:8501`.

Keep both terminals running while using the application.

## Model Artifacts

The backend uses pre-trained model artifacts and preprocessing components stored in `backend/artifacts/`, including:

- Logistic regression model for side-effect classification.
- Severity prediction model.
- Structured-data preprocessing pipeline.
- TF-IDF vectorizer for text features.
- Drug catalog and lookup data.
- Configuration files defining preprocessing, target labels, and API schemas.

The saved artifacts allow the application to perform inference without retraining the models on each startup.

## Notebooks

The `notebook/` directory contains notebooks for exploratory data analysis, preprocessing, and model development.

## Important Notes

- Model predictions are experimental and intended for educational and research purposes.
- Predictions should not be treated as medical advice or used as the sole basis for clinical decisions.
- Always consult a qualified healthcare professional regarding drug interactions and medication safety.
- Ensure all required model artifacts and dependencies are available before starting the application.

## Author

**Aya Saleh**

GitHub: [@aya-saleh123](https://github.com/aya-saleh123)
