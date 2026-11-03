"""Lab 13 - Full MLOps lifecycle in one command (the same sequence the GitHub Actions
workflow .github/workflows/mlops_pipeline.yml runs on every push / weekly schedule).
Use --fresh to wipe the local MLflow registry first, --docker to include Lab 9."""
import json
import os
import shutil
import subprocess
import sys
import time

STAGES = [
    ("Lab 5  data validation + preprocessing", "pipelines/run_lab5_pipeline.py"),
    ("Lab 6  train, register, promote",       "pipelines/run_lab6_registry.py"),
    ("Lab 7  export + serve",                  "pipelines/run_lab7_serving.py"),
    ("Lab 8  automated tests (quality gate)",  "pipelines/run_lab8_testing.py"),
    ("Lab 9  docker build + run",              "pipelines/run_lab9_docker.py"),
    ("Lab 10 drift detection",                 "pipelines/run_lab10_drift.py"),
    ("Lab 11 drift-triggered retraining",      "pipelines/run_lab11_retraining.py"),
    ("Lab 8  re-test the new champion",        "pipelines/run_lab8_testing.py"),
    ("Lab 12 production monitoring",           "pipelines/run_lab12_monitoring.py"),
]


def main():
    if "--fresh" in sys.argv:
        shutil.rmtree("mlruns", ignore_errors=True)
        if os.path.exists("mlflow.db"):
            os.remove("mlflow.db")
        print("[INFO] Local MLflow registry reset.")

    summary = []
    for name, script in STAGES:
        if "docker" in script and "--docker" not in sys.argv:
            summary.append({"stage": name, "status": "SKIPPED"})
            continue
        print(f"\n{'#' * 60}\n# {name}\n{'#' * 60}")
        t0 = time.time()
        rc = subprocess.run([sys.executable, script]).returncode
        summary.append({"stage": name, "status": "PASSED" if rc == 0 else "FAILED",
                        "seconds": round(time.time() - t0, 1)})
        if rc != 0:
            break

    os.makedirs("artifacts", exist_ok=True)
    with open("artifacts/end_to_end_summary.json", "w") as f:
        json.dump(summary, f, indent=4)
    print(f"\n{'=' * 60}\nEND-TO-END SUMMARY")
    for s in summary:
        print(f"  {s['status']:8} {s['stage']}  {s.get('seconds', '')}")
    ok = all(s["status"] != "FAILED" for s in summary)
    print("[SUCCESS] Full MLOps lifecycle complete!" if ok else "[ERROR] Lifecycle stopped at a failed stage.")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
