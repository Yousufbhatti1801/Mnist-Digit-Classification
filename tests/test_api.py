"""
API + predictor integration tests. Skipped automatically if no trained model
exists yet (run ``python -m scripts.train`` first). Uses FastAPI's TestClient,
so no server needs to be running.

    python -m pytest tests/test_api.py
"""

from __future__ import annotations

import numpy as np
import pytest

from mnist_classifier.config import load_config
from mnist_classifier.data import load_mnist

pytestmark = pytest.mark.skipif(
    not load_config().model_path.is_file(),
    reason="no trained model; run python -m scripts.train first",
)


@pytest.fixture(scope="module")
def client():
    from fastapi.testclient import TestClient

    from api.main import app

    with TestClient(app) as c:  # triggers the lifespan model load
        yield c


@pytest.fixture(scope="module")
def test_digit():
    cfg = load_config()
    _, _, Xte, yte = load_mnist(cfg.data_dir)
    return Xte[0], int(yte[0])


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["model_loaded"] is True


def test_predict_canvas_correct(client, test_digit):
    img, true = test_digit
    pixels = (img.astype(float) / 255.0).flatten().tolist()
    r = client.post("/predict/canvas", json={"pixels": pixels, "auto_orient": False})
    assert r.status_code == 200
    body = r.json()
    assert body["digit"] == true
    assert 0.0 <= body["confidence"] <= 1.0
    assert len(body["probabilities"]) == 10


def test_predict_canvas_rejects_non_square(client):
    r = client.post("/predict/canvas", json={"pixels": [0.0, 1.0, 0.5]})
    assert r.status_code == 422


def test_predict_upload_correct(client, test_digit):
    from PIL import Image
    import io

    img, true = test_digit
    buf = io.BytesIO()
    Image.fromarray(img).resize((140, 140)).save(buf, format="PNG")
    buf.seek(0)
    r = client.post("/predict", files={"file": ("d.png", buf, "image/png")})
    assert r.status_code == 200
    assert r.json()["digit"] == true
