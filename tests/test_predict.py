from fastapi.testclient import TestClient
from app import app

client = TestClient(app)

def test_predict_ok():
    payload = {"sepal_length":5.1,"sepal_width":3.5,"petal_length":1.4,"petal_width":0.2}
    r = client.post("/predict", json=payload)
    assert r.status_code == 200
    data = r.json()
    assert "predicted_class_label" in data
    assert set(data["probabilities"].keys()) >= {"setosa","versicolor","virginica"}
