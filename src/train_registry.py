import os
import sys
import numpy as np
import common
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

def train_and_register_model():
    print("[INFO] --- Starting MLflow Run with Model Registry ---")
    
    # 1. Set up MLflow
    mlflow.set_experiment("Bank_Marketing_Prediction")
    
    # 2. Load Processed Data
    try:
        X_train = np.load('data/processed/X_train_final.npy')
        X_test = np.load('data/processed/X_test_final.npy')
        y_train = np.load('data/processed/y_train.npy')
        y_test = np.load('data/processed/y_test.npy')
    except FileNotFoundError:
        print("[ERROR] Processed data not found. Please run the Lab 5 pipeline first.")
        return
    
    # 3. Define Model Parameters (pass "v1" or "v2" on the command line; default v2)
    versions = {
        "v1": {"n_estimators": 100, "max_depth": 10, "random_state": 42, "class_weight": "balanced"},
        "v2": {"n_estimators": 200, "max_depth": 15, "random_state": 42, "class_weight": "balanced"},
    }
    tag = (sys.argv[1] if len(sys.argv) > 1 else "v2").lower()
    params = versions[tag]

    with mlflow.start_run(run_name=f"RandomForest_Registry_{tag.upper()}") as run:
        # Log parameters
        mlflow.log_params(params)
        mlflow.log_param("model_family", "RandomForest")
        
        # 4. Train Model
        print("[INFO] Training RandomForest model...")
        rf_model = RandomForestClassifier(**params)
        rf_model.fit(X_train, y_train)
        
        # 5. Evaluate
        print("[INFO] Evaluating model...")
        y_pred = rf_model.predict(X_test)
        y_proba = rf_model.predict_proba(X_test)[:, 1]
        
        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "f1_score": f1_score(y_test, y_pred),
            "roc_auc": roc_auc_score(y_test, y_proba)
        }
        
        mlflow.log_metrics(metrics)
        
        # 6. LOG PREPROCESSOR PIPELINE (Lineage)
        print("[INFO] Attaching preprocessing pipeline to model artifacts...")
        mlflow.log_artifact("models/preprocessor.pkl", artifact_path="preprocessing_pipeline")
        
        # 7. LOG AND REGISTER THE MODEL
        print("[INFO] Pushing model to MLflow Registry...")
        mlflow.sklearn.log_model(
            sk_model=rf_model,
            # artifact_path="random_forest_model",
            # artifact_path is deprecated in newer MLflow versions
            name="random_forest_model",
            serialization_format="cloudpickle",  # newer MLflow defaults to skops, which rejects tree models
            registered_model_name="Bank_Marketing_Production_Model" # Creates the registry entry
        )
        
        print(f"[SUCCESS] Metrics logged: F1 = {metrics['f1_score']:.4f} | ROC-AUC = {metrics['roc_auc']:.4f}")
        print(f"[SUCCESS] Model successfully registered under name: 'Bank_Marketing_Production_Model'")

if __name__ == "__main__":
    train_and_register_model()