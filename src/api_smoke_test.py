"""Lab 7 - Smoke-test the API in-process (no server needed) with real test-set customers."""
import os
import sys
import json

sys.path.insert(0, os.getcwd())
from fastapi.testclient import TestClient
from app.main import app
from common import load_clean_raw, split_raw


def run_smoke_test():
    client = TestClient(app)
    _, X_test, _, y_test = split_raw(load_clean_raw())
    sample = X_test.head(5).to_dict("records")

    checks = {}
    checks["health"] = client.get("/health").json()["status"] == "ok"
    info = client.get("/model-info")
    checks["model_info"] = info.status_code == 200

    single = client.post("/predict", json=sample[0])
    checks["predict"] = single.status_code == 200 and 0 <= single.json()["probability"] <= 1

    batch = client.post("/predict/batch", json=sample)
    checks["predict_batch"] = batch.status_code == 200 and batch.json()["count"] == 5

    bad = dict(sample[0], age=-3)
    checks["rejects_invalid_input"] = client.post("/predict", json=bad).status_code == 422

    for k, v in checks.items():
        print(f"  {'PASS' if v else 'FAIL'}  {k}")
    print("\nExample request :", json.dumps(sample[0]))
    print("Example response:", single.json())

    os.makedirs("artifacts", exist_ok=True)
    with open("artifacts/api_smoke_test.json", "w") as f:
        json.dump({"checks": checks, "model_info": info.json(), "example_response": single.json()}, f, indent=4)
    return all(checks.values())


if __name__ == "__main__":
    ok = run_smoke_test()
    print("[SUCCESS] API smoke test passed." if ok else "[ERROR] API smoke test failed.")
    raise SystemExit(0 if ok else 1)
