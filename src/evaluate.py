import os
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

def run_evaluation():
    print("Starting Model Evaluation...")
    
    # 1. Load test data and the trained model
    X_test_final = np.load('data/processed/X_test_final.npy')
    y_test = np.load('data/processed/y_test.npy')
    model = joblib.load('models/random_forest_baseline.pkl')
    
    # 2. Make predictions
    y_pred = model.predict(X_test_final)
    y_prob = model.predict_proba(X_test_final)[:, 1]
    
    # 3. Calculate metrics
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)
    
    print("\n--- Model Evaluation Report ---")
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"F1-Score : {f1:.4f}")
    print(f"ROC-AUC  : {auc:.4f}")
    print("-------------------------------\n")
    
    # 4. Error Analysis Export
    # Reload the raw rows and repeat the exact preprocessing steps (dedupe, y -> 0/1,
    # same split parameters) so the test rows line up with y_test / y_pred.
    df = pd.read_csv('data/raw/bank-additional-full.csv', sep=';').drop_duplicates().reset_index(drop=True)
    y_all = df['y'].apply(lambda v: 1 if str(v).strip().lower() == 'yes' else 0).astype(int)
    _, X_test_raw, _, y_test_check = train_test_split(
        df.drop('y', axis=1), y_all, test_size=0.2, random_state=42, stratify=y_all
    )
    assert np.array_equal(y_test_check.to_numpy(), y_test), \
        "Raw test rows do not match processed y_test. Re-run preprocessing."
    
    errors_df = X_test_raw.copy()
    errors_df['Actual_Subscription'] = y_test
    errors_df['Predicted_Subscription'] = y_pred
    
    # False negatives: customers who WOULD have subscribed but were predicted not to
    false_negatives = errors_df[(errors_df['Actual_Subscription'] == 1) & (errors_df['Predicted_Subscription'] == 0)]
    # False positives: customers predicted to subscribe who did not
    false_positives = errors_df[(errors_df['Actual_Subscription'] == 0) & (errors_df['Predicted_Subscription'] == 1)]
    
    os.makedirs('outputs', exist_ok=True)
    false_negatives.to_csv('outputs/false_negatives.csv', index=False)
    false_positives.to_csv('outputs/false_positives.csv', index=False)
    
    print(f"False negatives: {len(false_negatives)} | False positives: {len(false_positives)}")
    print("Evaluation complete and error analysis files saved!")

if __name__ == "__main__":
    run_evaluation()
