import pytest
from fastapi.testclient import TestClient

from emotion_classification.api import create_app
from emotion_classification.config import id2label


@pytest.fixture
def client(trained_checkpoint, monkeypatch):
    monkeypatch.setenv("EMOTION_CHECKPOINT", trained_checkpoint)
    with TestClient(create_app()) as client:
        yield client


def test_predict_returns_an_emotion_name(client):
    response = client.post("/predict", json={"text": "i feel so happy today"})

    assert response.status_code == 200
    assert response.json()["prediction"] in id2label.values()


def test_predict_proba_returns_a_probability_per_emotion(client):
    response = client.post("/predict_proba", json={"text": "i feel so happy today"})

    assert response.status_code == 200
    probabilities = response.json()["probabilities"]
    assert set(probabilities) == set(id2label.values())
    assert sum(probabilities.values()) == pytest.approx(1.0, abs=1e-4)


def test_predict_proba_is_consistent_with_predict(client):
    text = {"text": "i feel so sad today"}

    prediction = client.post("/predict", json=text).json()["prediction"]
    probabilities = client.post("/predict_proba", json=text).json()["probabilities"]

    assert max(probabilities, key=probabilities.get) == prediction


def test_a_very_long_text_is_accepted(client):
    response = client.post("/predict", json={"text": "i feel so happy " * 500})

    assert response.status_code == 200


def test_the_checkpoint_argument_takes_precedence_over_the_environment(trained_checkpoint, monkeypatch):
    monkeypatch.setenv("EMOTION_CHECKPOINT", "/nonexistent/checkpoint")

    with TestClient(create_app(checkpoint=trained_checkpoint)) as client:
        response = client.post("/predict", json={"text": "i feel so happy today"})

    assert response.status_code == 200
