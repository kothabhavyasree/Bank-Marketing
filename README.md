# Bank Marketing Campaign Prediction

An end-to-end Machine Learning and MLOps project that predicts whether a customer will subscribe to a bank term deposit (`yes` or `no`) based on customer and campaign information.

**Project:** Bank Marketing  
**Dataset:** Bank Marketing additional dataset  
**Report:** `Bank_marketing.pdf`  
**Exploration notebook:** `notebooks/Bank_Marketing_Project_Final.ipynb`

## 1. Project Overview

The objective is to build a machine learning system that predicts whether a customer will subscribe to a bank term deposit.

This project demonstrates the complete ML lifecycle, including:

- Data validation and preprocessing
- Model training and evaluation
- MLflow experiment tracking
- Model registration and lifecycle management
- Model deployment using FastAPI
- Automated testing
- Docker containerization
- Data drift detection
- Drift-triggered model retraining
- Production prediction monitoring
- CI/CD/CT automation using GitHub Actions

## 2. Technology Stack

- **Programming:** Python
- **Data processing:** Pandas, NumPy
- **Machine Learning:** Scikit-learn
- **Experiment tracking and model registry:** MLflow
- **API development:** FastAPI, Uvicorn
- **Testing:** Pytest
- **Containerization:** Docker, Docker Compose
- **Automation:** GitHub Actions
- **Visualization:** Matplotlib

## 3. Project Structure

```text
Bank-Marketing/
│
├── .github/
│   └── workflows/
│       ├── ci.yml
│       └── mlops_pipeline.yml
│
├── app/
│   └── main.py
│
├── configs/
│   └── config.yaml
│
├── data/
│   ├── raw/
│   └── processed/
│
├── models/
├── notebooks/
├── pipelines/
├── src/
├── tests/
├── artifacts/
├── logs/
├── outputs/
│
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── requirements-ci.txt
├── requirements-serve.txt
└── README.md
```

## 4. Setup and Installation

Clone the repository:

```bash
git clone https://github.com/kothabhavyasree/Bank-Marketing.git
cd Bank-Marketing
```

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

For Windows:

```bash
.venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Place the required dataset in:

```text
data/raw/bank-additional-full.csv
```

## 5. Running the Labs

Run commands from the project root directory.

| Lab | Description | Command |
|---|---|---|
| 3 | Baseline ML pipeline | `python pipelines/run_lab3_baseline.py` |
| 4 | MLflow experiment tracking | `python pipelines/run_lab4_tracking.py` |
| 5 | Production data pipeline | `python pipelines/run_lab5_pipeline.py` |
| 6 | Model registry and lifecycle | `python pipelines/run_lab6_registry.py` |
| 7 | Model serving using FastAPI | `python pipelines/run_lab7_serving.py` |
| 8 | Automated testing | `python pipelines/run_lab8_testing.py` |
| 9 | Docker deployment | `python pipelines/run_lab9_docker.py` |
| 10 | Data drift detection | `python pipelines/run_lab10_drift.py` |
| 11 | Drift-triggered retraining | `python pipelines/run_lab11_retraining.py` |
| 12 | Production monitoring | `python pipelines/run_lab12_monitoring.py` |
| 13 | End-to-end MLOps pipeline | `python pipelines/run_lab13_end_to_end.py --fresh` |

To include Docker in the end-to-end pipeline:

```bash
python pipelines/run_lab13_end_to_end.py --fresh --docker
```

Labs should be run in sequence because later labs depend on outputs from earlier stages.

## 6. Model Serving – FastAPI

Start the API:

```bash
uvicorn app.main:app --reload
```

Open the API documentation:

```text
http://127.0.0.1:8000/docs
```

Available endpoints:

- `GET /health` – Check API health.
- `GET /model-info` – View production model information.
- `POST /predict` – Predict for one customer.
- `POST /predict/batch` – Predict for multiple customers.
- `POST /feedback` – Submit prediction feedback.

## 7. Docker Deployment

Make sure Docker Desktop is running.

Build and start the application:

```bash
docker compose up --build
```

Stop the application:

```bash
docker compose down
```

The Docker image packages the API and its serving dependencies.

## 8. Data Drift and Monitoring

The project demonstrates data drift detection using simulated customer data.

- **Lab 10:** Simulates a new customer batch and detects changes using statistical drift checks, including Population Stability Index (PSI) and KS tests.
- **Lab 11:** Retrains a challenger model when drift conditions require retraining and evaluates it against the existing production model.
- **Lab 12:** Generates prediction traffic and feedback, monitors prediction behaviour, and creates monitoring reports and visualizations.

The simulated drift is intended to demonstrate the MLOps workflow; it is not evidence of actual real-world customer drift.

## 9. Automated Testing and GitHub Actions

The project includes automated tests using Pytest.

Run tests locally:

```bash
python -m pytest tests -v
```

Two GitHub Actions workflows are included:

**CI workflow – `ci.yml`**

- Runs on pushes and pull requests.
- Installs the required CI dependencies.
- Builds required model artifacts.
- Executes automated tests.
- Uploads test results.

**MLOps workflow – `mlops_pipeline.yml`**

- Runs on selected pushes to the main branch.
- Supports scheduled execution.
- Supports manual execution through GitHub Actions.
- Runs the end-to-end MLOps lifecycle.
- Builds and smoke-tests the Docker image.
- Uploads model and report artifacts.

## 10. Project-Specific Decisions

- Exact duplicate records are removed before splitting.
- The `duration` feature is dropped because it is only known after the call and could cause data leakage.
- Unknown categorical values are retained as a separate category.
- The dataset is split using an 80/20 stratified split with `random_state=42`.
- One-hot encoding and standard scaling are used for preprocessing.
- Class imbalance is addressed using `class_weight="balanced"`.
- Recall is used as an important model-selection metric because identifying potential subscribers is a key objective.
- Drifted data is simulated for demonstrating the retraining and monitoring workflow.

## 11. Model Results

In the verified local MLflow run, Production Model Version 1 achieved:

| Metric | Value |
|---|---:|
| F1-score | 0.4969 |
| ROC-AUC | 0.8132 |
| Recall | 0.6401 |

The challenger model was evaluated against the existing production model. It did not meet the promotion condition, so Version 1 remained in Production.

## 12. Conclusion

This project demonstrates how a traditional machine learning model can be developed into an automated MLOps application.

It covers the complete workflow from raw data validation and model training to model deployment, Docker, drift detection, retraining, monitoring, and automated CI/CD/CT pipelines.

The project provides practical experience in building, testing, deploying, and monitoring machine learning applications.