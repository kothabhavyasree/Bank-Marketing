import os
import json
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder

def run_preprocessing():
    print("Starting Preprocessing Pipeline...")
    
    os.makedirs('data/processed', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    
    # 1. Load the raw data (semicolon-separated UCI file)
    data_path = 'data/raw/bank-additional-full.csv'
    df = pd.read_csv(data_path, sep=';')
    
    # 2. Data Cleaning
    # Remove exact duplicates (12 rows in the original dataset)
    df = df.drop_duplicates().reset_index(drop=True)
    
    # 'duration' is only known after the call, so it is dropped (target leakage)
    df = df.drop('duration', axis=1)
    
    # Unconditional string to binary integer conversion (no = 0, yes = 1)
    df['y'] = df['y'].apply(lambda v: 1 if str(v).strip().lower() == 'yes' else 0).astype(int)
        
    X = df.drop('y', axis=1)
    y = df['y']
    
    # 3. Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # 4. Separate Column Types
    num_cols = X_train.select_dtypes(include='number').columns.tolist()
    cat_cols = [c for c in X_train.columns if c not in num_cols]
    
    # 5. Scale & Encode
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(X_train[num_cols])
    x_test_scaled = scaler.transform(X_test[num_cols])
    
    ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    x_train_encoded = ohe.fit_transform(X_train[cat_cols])
    x_test_encoded = ohe.transform(X_test[cat_cols])
    
    # Combine Features
    X_train_final = np.hstack((x_train_scaled, x_train_encoded))
    X_test_final = np.hstack((x_test_scaled, x_test_encoded))
    
    # 6. Save Artifacts with explicit integer type casting
    np.save('data/processed/X_train_final.npy', X_train_final)
    np.save('data/processed/X_test_final.npy', X_test_final)
    np.save('data/processed/y_train.npy', y_train.to_numpy(dtype=np.int64))
    np.save('data/processed/y_test.npy', y_test.to_numpy(dtype=np.int64))
    
    joblib.dump(scaler, 'models/scaler.pkl')
    joblib.dump(ohe, 'models/encoder.pkl')
    
    # Save Metadata
    metadata = {
        "dataset_name": "Bank Marketing (bank-additional-full)",
        "target": "y",
        "dropped_features": ["duration"],
        "train_shape": list(X_train_final.shape),
        "test_shape": list(X_test_final.shape),
        "numerical_features": num_cols,
        "categorical_features": cat_cols
    }
    with open('data/processed/dataset_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=4)
        
    print(f"Train shape: {X_train_final.shape} | Test shape: {X_test_final.shape}")
    print("Preprocessing completed successfully!")

if __name__ == "__main__":
    run_preprocessing()
