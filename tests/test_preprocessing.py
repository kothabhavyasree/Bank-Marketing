import joblib
import numpy as np
from conftest import require


def test_preprocessor_output_shape(raw_split):
    require("models/preprocessor.pkl")
    pre = joblib.load("models/preprocessor.pkl")
    X = pre.transform(raw_split[1].head(50))
    assert X.shape == (50, 62)
    assert not np.isnan(X).any()


def test_preprocessor_handles_unseen_category(raw_split):
    require("models/preprocessor.pkl")
    pre = joblib.load("models/preprocessor.pkl")
    row = raw_split[1].head(1).copy()
    row["job"] = "brand-new-job"        # handle_unknown='ignore' must not crash
    assert pre.transform(row).shape == (1, 62)
