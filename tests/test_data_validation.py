import pandas as pd
from validate_data import validate_schema

RAW = "data/raw/bank-additional-full.csv"


def test_raw_data_passes_schema():
    assert validate_schema(pd.read_csv(RAW, sep=";"), "test_schema_ok.csv")


def test_schema_catches_corrupted_rows():
    df = pd.read_csv(RAW, sep=";", nrows=200)
    df.loc[0, "age"] = 250              # out of range
    df.loc[1, "job"] = "astronaut"      # unknown category
    assert not validate_schema(df, "test_schema_bad.csv")


def test_split_is_stratified_and_deterministic(raw_split):
    X_train, X_test, y_train, y_test = raw_split
    assert len(X_train) == 32940 and len(X_test) == 8236
    assert abs(y_train.mean() - y_test.mean()) < 0.005
    assert "duration" not in X_train.columns
