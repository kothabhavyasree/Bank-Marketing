import pytest
from conftest import require


@pytest.fixture(scope="module")
def client(cfg):
    require(cfg["serving"]["inference_pipeline"])
    from fastapi.testclient import TestClient
    from app.main import app
    return TestClient(app)


@pytest.fixture(scope="module")
def customer(raw_split):
    return raw_split[1].head(1).to_dict("records")[0]


def test_health(client):
    assert client.get("/health").json()["status"] == "ok"


def test_predict(client, customer):
    r = client.post("/predict", json=customer)
    assert r.status_code == 200
    assert set(r.json()) == {"prediction_id", "probability", "prediction", "label"}


def test_batch(client, raw_split):
    rows = raw_split[1].head(10).to_dict("records")
    assert client.post("/predict/batch", json=rows).json()["count"] == 10


@pytest.mark.parametrize("field,value", [("age", 5), ("contact", "email"), ("campaign", 0)])
def test_invalid_input_rejected(client, customer, field, value):
    assert client.post("/predict", json=dict(customer, **{field: value})).status_code == 422


def test_missing_field_rejected(client, customer):
    bad = {k: v for k, v in customer.items() if k != "euribor3m"}
    assert client.post("/predict", json=bad).status_code == 422
