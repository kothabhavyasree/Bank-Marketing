"""Shared helpers for Labs 7-13 (config, raw data loading, the fixed train/test split)."""
import os
import mlflow
import yaml
import pandas as pd
from sklearn.model_selection import train_test_split


# Use the MLflow database inside the actual project directory
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MLFLOW_DB = os.path.join(PROJECT_ROOT, "mlflow.db")
mlflow.set_tracking_uri("sqlite:///" + MLFLOW_DB)

def load_config(path="configs/config.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)


def load_clean_raw(path="data/raw/bank-additional-full.csv"):
    """Same cleaning as preprocess_pipeline.py: dedupe, drop leaky 'duration', y -> 0/1."""
    df = pd.read_csv(path, sep=";").drop_duplicates().reset_index(drop=True)
    df = df.drop(columns=["duration"], errors="ignore")
    df["y"] = (df["y"].astype(str).str.strip().str.lower() == "yes").astype(int)
    return df


def split_raw(df):
    """Identical split to Lab 5 (80/20, stratified, random_state=42)."""
    X, y = df.drop(columns="y"), df["y"]
    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


def build_preprocessor(num_cols, cat_cols):
    """Same ColumnTransformer as Lab 5 (median-impute + scale / mode-impute + one-hot)."""
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
    num = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])
    cat = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                    ("ohe", OneHotEncoder(sparse_output=False, handle_unknown="ignore"))])
    return ColumnTransformer([("num", num, num_cols), ("cat", cat, cat_cols)])


def psi(reference, current, bins=10):
    """Population Stability Index. <0.1 stable, 0.1-0.2 moderate, >0.2 significant drift."""
    import numpy as np
    import pandas as pd
    if pd.api.types.is_numeric_dtype(reference) and reference.nunique() > bins:
        edges = np.unique(np.quantile(reference, np.linspace(0, 1, bins + 1)))
        edges[0], edges[-1] = -np.inf, np.inf
        ref = pd.cut(reference, edges).value_counts(normalize=True, sort=False)
        cur = pd.cut(current, edges).value_counts(normalize=True, sort=False)
    else:
        cats = sorted(set(reference.astype(str)) | set(current.astype(str)))
        ref = reference.astype(str).value_counts(normalize=True).reindex(cats, fill_value=0)
        cur = current.astype(str).value_counts(normalize=True).reindex(cats, fill_value=0)
    ref, cur = np.clip(ref.values, 1e-4, None), np.clip(cur.values, 1e-4, None)
    return float(np.sum((cur - ref) * np.log(cur / ref)))
