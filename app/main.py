"""Lab 7 - FastAPI service for the Bank Marketing model.
Run from the project root:  uvicorn app.main:app --reload   ->  http://127.0.0.1:8000/docs
"""
import os
import json
import time
import uuid
from datetime import datetime, timezone
from typing import List, Literal

import joblib
import pandas as pd
import yaml
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field

with open(os.getenv("CONFIG_PATH", "configs/config.yaml")) as f:
    CFG = yaml.safe_load(f)["serving"]

YesNoUnknown = Literal["yes", "no", "unknown"]


class Customer(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    age: int = Field(..., ge=17, le=100)
    job: Literal["admin.", "blue-collar", "entrepreneur", "housemaid", "management", "retired",
                 "self-employed", "services", "student", "technician", "unemployed", "unknown"]
    marital: Literal["divorced", "married", "single", "unknown"]
    education: Literal["basic.4y", "basic.6y", "basic.9y", "high.school", "illiterate",
                       "professional.course", "university.degree", "unknown"]
    default: YesNoUnknown
    housing: YesNoUnknown
    loan: YesNoUnknown
    contact: Literal["cellular", "telephone"]
    month: Literal["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"]
    day_of_week: Literal["mon", "tue", "wed", "thu", "fri"]
    campaign: int = Field(..., ge=1)
    pdays: int = Field(..., ge=0, le=999)
    previous: int = Field(..., ge=0)
    poutcome: Literal["failure", "nonexistent", "success"]
    emp_var_rate: float = Field(..., alias="emp.var.rate")
    cons_price_idx: float = Field(..., alias="cons.price.idx")
    cons_conf_idx: float = Field(..., alias="cons.conf.idx")
    euribor3m: float
    nr_employed: float = Field(..., alias="nr.employed")


class Feedback(BaseModel):
    prediction_id: str
    actual: int = Field(..., ge=0, le=1)


app = FastAPI(title="Bank Marketing Subscription API", version="1.0")
_state = {}


def get_model():
    if "pipeline" not in _state:
        path = CFG["inference_pipeline"]
        if not os.path.exists(path):
            raise HTTPException(503, "Model not exported yet. Run pipelines/run_lab7_serving.py")
        _state["pipeline"] = joblib.load(path)
        with open(CFG["model_info"]) as f:
            _state["info"] = json.load(f)
    return _state["pipeline"], _state["info"]


def _log(record):
    os.makedirs(os.path.dirname(CFG["prediction_log"]), exist_ok=True)
    with open(CFG["prediction_log"], "a") as f:
        f.write(json.dumps(record) + "\n")


def _predict(customers: List[Customer]):
    pipeline, info = get_model()
    start = time.perf_counter()
    df = pd.DataFrame([c.model_dump(by_alias=True) for c in customers])[info["features"]]
    probs = pipeline.predict_proba(df)[:, 1]
    latency_ms = (time.perf_counter() - start) * 1000 / len(customers)

    results = []
    for row, p in zip(df.to_dict("records"), probs):
        pred = int(p >= CFG["threshold"])
        rec = {"prediction_id": str(uuid.uuid4()), "probability": round(float(p), 4),
               "prediction": pred, "label": "yes" if pred else "no"}
        _log({**rec, "timestamp": datetime.now(timezone.utc).isoformat(),
              "model_version": info["version"], "latency_ms": round(latency_ms, 3), "features": row})
        results.append(rec)
    return results


@app.get("/health")
def health():
    ready = os.path.exists(CFG["inference_pipeline"])
    return {"status": "ok" if ready else "model_missing", "model_loaded": "pipeline" in _state}


@app.get("/model-info")
def model_info():
    _, info = get_model()
    return {k: info[k] for k in ("model_name", "version", "run_id", "metrics")}


@app.post("/predict")
def predict(customer: Customer):
    return _predict([customer])[0]


@app.post("/predict/batch")
def predict_batch(customers: List[Customer]):
    if not customers:
        raise HTTPException(400, "Empty batch")
    return {"count": len(customers), "predictions": _predict(customers)}


@app.post("/feedback")
def feedback(fb: Feedback):
    """Ground truth arrives later (did the customer subscribe?) - used by Lab 12 monitoring."""
    _log({"type": "feedback", "prediction_id": fb.prediction_id, "actual": fb.actual,
          "timestamp": datetime.now(timezone.utc).isoformat()})
    return {"status": "recorded"}
