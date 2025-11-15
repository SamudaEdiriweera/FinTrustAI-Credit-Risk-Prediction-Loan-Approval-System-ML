# FinTrustAI: End-to-End Loan Approval Prediction System

![Project Banner](https://placehold.co/1200x400/282c34/61dafb?text=FinTrustAI%20Loan%20Prediction)

**A complete, production-ready Machine Learning project with a full MLOps pipeline for predicting loan approval status. This repository demonstrates best practices in model development, API deployment, and CI/CD automation.**

---

## Table of Contents
- [Project Overview](#project-overview)
- [Features](#features)
- [Tech Stack](#tech-stack)
- [MLOps Pipeline Architecture](#mlops-pipeline-architecture)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation & Setup](#installation--setup)
- [How to Run](#how-to-run)
  - [1. Data Preprocessing & Training](#1-data-preprocessing--training)
  - [2. Hyperparameter Tuning](#2-hyperparameter-tuning)
  - [3. Run the Backend API Service](#3-run-the-backend-api-service)
- [Automated CI/CD with GitHub Actions](#automated-cicd-with-github-actions)
- [Future Work](#future-work)
- [Contact](#contact)

---

## Project Overview

FinTrustAI is a system designed to automate the loan approval process by predicting whether a loan application will be approved based on the applicant's profile. The project goes beyond a simple model in a notebook; it implements a full-stack application with a robust MLOps workflow, making it scalable, reproducible, and ready for deployment.

The core of the project is a classification model (initially `Logistic Regression`) that is trained on historical loan application data. This model is served via a FastAPI backend and can be interacted with through a Next.js frontend.

---

## Features

- **End-to-End ML Workflow**: From data preprocessing and model training to API deployment.
- **Comparative Hyperparameter Tuning**: Scripts to compare `GridSearchCV`, `RandomizedSearchCV`, and `Optuna` to find the best model parameters.
- **Experiment Tracking**: Integrated with **MLflow** to log parameters, metrics, and model artifacts for every run.
- **RESTful API**: A **FastAPI** backend serves the trained model, making predictions available via a `/predict` endpoint.
- **Interactive Frontend**: A **Next.js** (TypeScript) user interface to interact with the model in real-time.
- **Containerization**: Both backend and frontend services are containerized with **Docker** for consistency and portability.
- **CI/CD Automation**: **GitHub Actions** workflows automate testing, building, and pushing Docker images for both services.
- **Modular & Configurable Code**: A clean, organized codebase with a central configuration file for easy management.

---

## Tech Stack

| Category      | Technology                                                                                                   |
|---------------|--------------------------------------------------------------------------------------------------------------|
| **Backend**   | ![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white) ![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=for-the-badge&logo=fastapi)           |
| **Frontend**  | ![Next.js](https://img.shields.io/badge/Next-black?style=for-the-badge&logo=next.js&logoColor=white) ![TypeScript](https://img.shields.io/badge/TypeScript-007ACC?style=for-the-badge&logo=typescript&logoColor=white) ![React](https://img.shields.io/badge/react-%2320232a.svg?style=for-the-badge&logo=react&logoColor=%2361DAFB) |
| **ML/Data**   | ![Scikit-learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white) ![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white) ![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white) |
| **MLOps**     | ![Docker](https://img.shields.io/badge/docker-%230db7ed.svg?style=for-the-badge&logo=docker&logoColor=white) ![GitHub Actions](https://img.shields.io/badge/github%20actions-%232671E5.svg?style=for-the-badge&logo=githubactions&logoColor=white) ![MLflow](https://img.shields.io/badge/mlflow-%230194E2.svg?style=for-the-badge&logo=mlflow&logoColor=white) |
| **Tooling**   | ![Git](https://img.shields.io/badge/git-%23F05033.svg?style=for-the-badge&logo=git&logoColor=white) ![Visual Studio Code](https://img.shields.io/badge/Visual%20Studio%20Code-0078d7.svg?style=for-the-badge&logo=visual-studio-code&logoColor=white) |

---

## MLOps Pipeline Architecture

This project implements an **MLOps Level 2** pipeline, focusing on CI/CD and automation.

![MLOps Pipeline Diagram](https://placehold.co/800x300/ffffff/000000?text=Your%20MLOps%20Workflow%20Diagram%20Here)


*(You can create a simple diagram using tools like diagrams.net and replace this placeholder)*

1.  **Development & Experimentation**: Data scientists work in notebooks, then modularize the code. All experiments are tracked with MLflow.
2.  **Code Commit (Git)**: Changes to backend code are pushed to the GitHub repository.
3.  **Continuous Integration (GitHub Actions)**:
    - A push triggers the relevant CI workflow (backend or frontend).
    - The workflow installs dependencies and runs automated tests (`pytest` for backend).
4.  **Continuous Delivery (GitHub Actions)**:
    - If tests pass, the workflow builds a Docker image for the service.
    - The new image is pushed to a container registry (e.g., Docker Hub).
5.  **Deployment**: (Manual for now) The container can be pulled from the registry and run anywhere.

---

## Project Structure

```bash
.
├── .github/workflows/                           # GitHub Actions CI/CD pipelines
│ ├── backend_ci_pipeline.yml
├── data/
│ └── Load_Data.csv                              # The dataset
├── src/                                         # Source code for the Python backend
│ ├── init.py
│ ├── config.py                                  # Central configuration file
│ ├── data_preprocessing.py
│ ├── model_evaluation.py
│ └── train.py
├── tests/
│ └── test_app.py                                # Pytests for the FastAPI backend
├── app.py                                       # FastAPI application script
├── Dockerfile                                   # Dockerfile for the backend
├── requirements.txt                             # Python dependencies
├── train_final_model.py                         # Script to train and save the best model
├── tune_*.py                                    # Scripts for hyperparameter tuning
└── README.md
```
---

## Getting Started

### Prerequisites

- [Python 3.8+](https://www.python.org/downloads/)
- [Node.js and npm](https://nodejs.org/en/download/) (for the frontend)
- [Docker Desktop](https://www.docker.com/products/docker-desktop/)
- An MLflow server (optional, for advanced tracking) or use local file storage.

### Installation & Setup

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/FinTrustAI-Credit-Risk-Prediction-Loan-Approval-System-ML.git
    cd FinTrustAI-Credit-Risk-Prediction-Loan-Approval-System-ML
    ```

2.  **Setup the Python Backend:**
    ```bash
    # Create and activate a virtual environment
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate

    # Install Python dependencies
    pip install -r requirements.txt
    ```

---

## How to Run

### 1. Data Preprocessing & Training
First, you need a trained model artifact. The `train_final_model.py` script uses the best parameters defined in `src/config.py` to train and save the model.

```bash
# From the root directory
python train_final_model.py
```
This will create loan_approval_classifier.joblib, which is required by the API.

### 2. Hyperparameter Tuning
To find the best model parameters, you can run the tuning scripts. This is an optional step if you just want to run the application.

```bash
# Example using Optuna (recommended)
python tune_optuna.py
```
After running, analyze the results with MLflow: mlflow ui.

### 3. Run the Backend API Service
You can run the FastAPI server either directly or using Docker (recommended).

**Option A: Directly with Uvicorn**
```bash
# From the root directory
uvicorn app:app --reload --port 8000
```
The API will be available at http://127.0.0.1:8000.

**Option B: Using Docker**
```bash
# Build the Docker image for the backend
docker build -t loan-predictor-backend .

# Run the container
docker run -p 8000:8000 loan-predictor-backend
```
## Automated CI/CD with GitHub Actions

This project is configured with a GitHub Actions workflows:

1.  **Backend CI Pipeline: **
    Triggered on pushes to backend-related files. It runs tests and builds the backend Docker image.

## 🚀 Future Work

The current model is a `Logistic Regression` baseline. The next steps will focus on improving model performance and expanding the MLOps pipeline.

- [ ] **Experiment with Advanced Models**:
  - [ ] Implement and tune a `Random Forest` classifier.
  - [ ] Implement and tune a `XGBoost` classifier.
  - [ ] Implement and tune a `LightGBM` classifier for performance comparison.

- [ ] **Advanced Feature Engineering**:
  - [ ] Create more complex features like `Debt-to-Income Ratio` and `Loan Term to Income` interactions.
  - [ ] Perform feature selection to identify the most impactful features.

- [ ] **Model Monitoring**:
  - [ ] Implement a system to monitor for *data drift* and *concept drift* after deployment using tools like Evidently AI.
  - [ ] Set up alerts for when model performance degrades below a certain threshold.

- [ ] **Full Continuous Deployment (CD) Pipeline**:
  - [ ] Add a final step to the GitHub Actions workflows to automatically deploy the containers to a cloud service (e.g., AWS Elastic Container Service, Azure App Service, Google Cloud Run).

- [ ] **Data Validation**:
  - [ ] Integrate a data validation framework like `Great Expectations` into the pipeline to ensure the quality and schema of incoming data before training and prediction.


