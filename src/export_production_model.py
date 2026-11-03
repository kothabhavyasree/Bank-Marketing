"""Lab 7 - Export the Production model or reuse a verified local pipeline."""

import os
import json
import joblib
import mlflow
import mlflow.sklearn
import common

from mlflow.tracking import MlflowClient
from sklearn.pipeline import Pipeline
from common import load_config


def export_production_model():
    cfg = load_config()
    name = cfg["project"]["registered_model"]
    pipeline_path = cfg["serving"]["inference_pipeline"]
    info_path = cfg["serving"]["model_info"]

    # Reuse the existing verified pipeline if MLflow artifacts are unavailable.
    def use_existing_pipeline():
        if not os.path.exists(pipeline_path) or not os.path.exists(info_path):
            return False

        try:
            pipeline = joblib.load(pipeline_path)

            with open(info_path) as f:
                info = json.load(f)

            if not isinstance(pipeline, Pipeline):
                return False

            if info.get("model_name") != name:
                return False

            if not {"preprocessor", "model"}.issubset(pipeline.named_steps):
                return False

            print("[SUCCESS] Reusing existing verified inference pipeline.")
            print(f"[INFO] Model version: {info.get('version')}")
            print(f"[INFO] Pipeline: {pipeline_path}")
            return True

        except Exception as e:
            print("[ERROR] Existing pipeline validation failed:", e)
            return False

    try:
        client = MlflowClient()
        versions = client.search_model_versions(f"name='{name}'")
        champion = next(
            (mv for mv in versions if mv.current_stage == "Production"),
            None
        )

        if champion is None:
            print("[WARNING] No Production model found in MLflow.")
            return use_existing_pipeline()

        print(f"[INFO] Loading {name} v{champion.version} from MLflow...")
        model = mlflow.sklearn.load_model(
            f"models:/{name}/{champion.version}"
        )

        pre_path = mlflow.artifacts.download_artifacts(
            run_id=champion.run_id,
            artifact_path="preprocessing_pipeline/preprocessor.pkl"
        )
        preprocessor = joblib.load(pre_path)

        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("model", model)
        ])

        os.makedirs("models", exist_ok=True)
        joblib.dump(pipeline, pipeline_path)

        run = client.get_run(champion.run_id)
        info = {
            "model_name": name,
            "version": int(champion.version),
            "run_id": champion.run_id,
            "metrics": run.data.metrics,
            "params": run.data.params,
            "features": list(preprocessor.feature_names_in_),
        }

        with open(info_path, "w") as f:
            json.dump(info, f, indent=4)

        print(f"[SUCCESS] Inference pipeline saved to {pipeline_path}")
        return True

    except Exception as e:
        print("[WARNING] MLflow export failed:", e)
        print("[INFO] Trying the existing local pipeline...")
        return use_existing_pipeline()


if __name__ == "__main__":
    raise SystemExit(0 if export_production_model() else 1)
