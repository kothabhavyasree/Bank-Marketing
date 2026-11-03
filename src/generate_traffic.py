"""Lab 12 - Simulate production traffic through the API: normal customers first, then the
drifted batch, sending ground-truth feedback for each prediction (as the bank would later)."""
import os
import sys

import pandas as pd

sys.path.insert(0, os.getcwd())
from fastapi.testclient import TestClient
from app.main import app, CFG
from common import load_config, load_clean_raw, split_raw


def generate(batch_size=50):
    cfg = load_config()
    if os.path.exists(CFG["prediction_log"]):
        os.remove(CFG["prediction_log"])          # fresh log for this monitoring run
    client = TestClient(app)

    _, X_test, _, y_test = split_raw(load_clean_raw())
    drifted = pd.read_csv(cfg["drift"]["current_batch"], sep=";")
    phases = [("normal", X_test.head(2000), y_test.head(2000).to_numpy()),
              ("drifted", drifted.drop(columns="y"), (drifted["y"] == "yes").astype(int).to_numpy())]

    total = 0
    for name, X, y in phases:
        rows = X.to_dict("records")
        for i in range(0, len(rows), batch_size):
            preds = client.post("/predict/batch", json=rows[i:i + batch_size]).json()["predictions"]
            for p, actual in zip(preds, y[i:i + batch_size]):
                client.post("/feedback", json={"prediction_id": p["prediction_id"], "actual": int(actual)})
        total += len(rows)
        print(f"[INFO] Sent {len(rows)} '{name}' requests")
    print(f"[SUCCESS] {total} predictions + feedback logged to {CFG['prediction_log']}")


if __name__ == "__main__":
    generate()
