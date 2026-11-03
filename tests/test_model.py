import joblib
from sklearn.metrics import recall_score, roc_auc_score
from conftest import require


def test_model_meets_quality_gate(cfg, raw_split):
    require(cfg["serving"]["inference_pipeline"])
    pipe = joblib.load(cfg["serving"]["inference_pipeline"])
    _, X_test, _, y_test = raw_split
    prob = pipe.predict_proba(X_test)[:, 1]
    recall = recall_score(y_test, (prob >= cfg["serving"]["threshold"]).astype(int))
    auc = roc_auc_score(y_test, prob)
    assert recall >= cfg["quality_gate"]["min_recall"], f"recall {recall:.3f} below gate"
    assert auc >= cfg["quality_gate"]["min_roc_auc"], f"ROC-AUC {auc:.3f} below gate"


def test_predictions_are_valid_probabilities(cfg, raw_split):
    require(cfg["serving"]["inference_pipeline"])
    pipe = joblib.load(cfg["serving"]["inference_pipeline"])
    prob = pipe.predict_proba(raw_split[1].head(100))
    assert prob.shape == (100, 2)
    assert ((prob >= 0) & (prob <= 1)).all()
