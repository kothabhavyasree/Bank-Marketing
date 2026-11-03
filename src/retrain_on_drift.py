"""Lab 11 - If Lab 10 flagged drift, retrain on (original train + new labelled batch),
compare with the current champion on old AND new data, and register the challenger.
The Lab 6 lifecycle manager then decides whether it gets promoted."""
import json
import os
import tempfile
import common
import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split

from common import build_preprocessor, load_config, load_clean_raw, split_raw


def scores(y, prob):
    pred = (prob >= 0.5).astype(int)
    return {"accuracy": accuracy_score(y, pred), "precision": precision_score(y, pred),
            "recall": recall_score(y, pred), "f1_score": f1_score(y, pred), "roc_auc": roc_auc_score(y, prob)}


def retrain():
    cfg = load_config()
    with open("artifacts/drift_report.json") as f:
        drift = json.load(f)
    if not drift["retrain_required"]:
        print("[INFO] No significant drift - retraining skipped.")
        return True

    # 1. Data: old training split + 70% of the new batch; 30% of the new batch held out
    X_train, X_test, y_train, y_test = split_raw(load_clean_raw())
    batch = pd.read_csv(cfg["drift"]["current_batch"], sep=";")
    batch["y"] = (batch["y"] == "yes").astype(int)
    new_tr, new_ho = train_test_split(batch, test_size=cfg["drift"]["holdout_fraction"],
                                      random_state=42, stratify=batch["y"])
    X_comb = pd.concat([X_train, new_tr.drop(columns="y")], ignore_index=True)
    y_comb = pd.concat([y_train, new_tr["y"]], ignore_index=True)
    X_ho, y_ho = new_ho.drop(columns="y"), new_ho["y"]

    # 2. Champion baseline (current Production pipeline)
    champion = joblib.load(cfg["serving"]["inference_pipeline"])
    with open(cfg["serving"]["model_info"]) as f:
        info = json.load(f)
    champ = {"original_test": scores(y_test, champion.predict_proba(X_test)[:, 1]),
             "drift_holdout": scores(y_ho, champion.predict_proba(X_ho)[:, 1])}

    # 3. Challenger: fresh preprocessor + same hyper-parameters as the champion
    num_cols = X_comb.select_dtypes(include="number").columns.tolist()
    cat_cols = [c for c in X_comb.columns if c not in num_cols]
    pre = build_preprocessor(num_cols, cat_cols)
    Xc = pre.fit_transform(X_comb)
    params = {k: int(v) if str(v).isdigit() else v for k, v in info["params"].items() if k != "model_family"}
    model = RandomForestClassifier(**params).fit(Xc, y_comb)
    chal = {"original_test": scores(y_test, model.predict_proba(pre.transform(X_test))[:, 1]),
            "drift_holdout": scores(y_ho, model.predict_proba(pre.transform(X_ho))[:, 1])}

    # 4. Log + register. 'recall' (the lifecycle metric) = recall on the original test set,
    #    so it is comparable with every earlier registry version.
    mlflow.set_experiment(cfg["project"]["experiment"])
    with mlflow.start_run(run_name="RandomForest_Retrained_Drift"), tempfile.TemporaryDirectory() as tmp:
        mlflow.log_params(params)
        mlflow.log_params({"model_family": "RandomForest", "trigger": "data_drift",
                           "train_rows": len(X_comb), "new_rows_added": len(new_tr)})
        mlflow.log_metrics(chal["original_test"])
        mlflow.log_metrics({f"drift_holdout_{k}": v for k, v in chal["drift_holdout"].items()})
        mlflow.log_artifact("artifacts/drift_report.json", artifact_path="drift")
        # Logged as preprocessing_pipeline/preprocessor.pkl so export_production_model.py picks it up
        joblib.dump(pre, os.path.join(tmp, "preprocessor.pkl"))
        mlflow.log_artifact(os.path.join(tmp, "preprocessor.pkl"), artifact_path="preprocessing_pipeline")
        mlflow.sklearn.log_model(model, name="random_forest_model", serialization_format="cloudpickle",
                                 registered_model_name=cfg["project"]["registered_model"])

    report = {"champion_version": info["version"], "champion": champ, "challenger": chal,
              "rows": {"combined_train": len(X_comb), "drift_holdout": len(X_ho)}}
    with open("artifacts/retraining_report.json", "w") as f:
        json.dump(report, f, indent=4)

    print(f"{'':16}{'champion':>12}{'challenger':>12}")
    for split in ("original_test", "drift_holdout"):
        for m in ("recall", "roc_auc"):
            print(f"{split[:9] + ' ' + m:16}{champ[split][m]:12.4f}{chal[split][m]:12.4f}")
    print("[SUCCESS] Challenger registered. Saved artifacts/retraining_report.json")
    return True


if __name__ == "__main__":
    raise SystemExit(0 if retrain() else 1)
