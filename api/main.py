"""
FastAPI backend for the MNIST digit classifier.

Endpoints
---------
GET  /health          liveness + whether the model is loaded
POST /predict         multipart image upload (PNG/JPG of any size) -> Prediction
POST /predict/canvas  JSON {pixels: [...], auto_orient} from the draw UI -> Prediction
GET  /                the draw-a-digit web UI (static HTML)

The trained Keras model is loaded exactly once at startup via the lifespan hook
and shared across requests (Keras inference is thread-safe for our use). All image
preprocessing lives in ``mnist_classifier.inference.Predictor`` so the API stays a
thin transport layer.

Run:
    api.main:appuvicorn  --reload          # dev, http://127.0.0.1:8000
    python -m api.main                     # same, via __main__
"""

from __future__ import annotations

import io
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from mnist_classifier import Predictor, load_config

from .schemas import CanvasRequest, HealthResponse, Prediction

UI_DIR = Path(__file__).resolve().parents[1] / "ui"

# Process-wide singletons populated at startup.
_state: dict = {"predictor": None, "config": None}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the model once before serving; fail loud if it is missing."""
    config = load_config()
    try:
        _state["predictor"] = Predictor.from_config(config)
        print(f"[api] model loaded from {config.model_path}")
    except FileNotFoundError as e:
        # Keep the app up so /health can report the problem, but predictions 503.
        print(f"[api] WARNING: {e}")
        _state["predictor"] = None
    _state["config"] = config
    yield
    _state.clear()


app = FastAPI(
    title="MNIST Digit Classifier API",
    description="Serve a Keras ANN that recognizes handwritten digits 0-9.",
    version="1.0.0",
    lifespan=lifespan,
)

# Permissive CORS so the static UI (or any front-end) can call the API in dev.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def _get_predictor() -> Predictor:
    predictor = _state.get("predictor")
    if predictor is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Train one first: python -m scripts.train",
        )
    return predictor


# --------------------------------------------------------------------------- #
# API endpoints
# --------------------------------------------------------------------------- #
@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    config = _state.get("config") or load_config()
    return HealthResponse(
        status="ok",
        model_loaded=_state.get("predictor") is not None,
        model_path=str(config.model_path),
        n_classes=config.n_classes,
    )


@app.post("/predict", response_model=Prediction)
async def predict_upload(file: UploadFile = File(...)) -> Prediction:
    """Predict the digit in an uploaded image (any size; grayscaled server-side)."""
    if not (file.content_type or "").startswith("image/"):
        raise HTTPException(status_code=400, detail="Upload must be an image.")
    raw = await file.read()
    try:
        from PIL import Image

        img = Image.open(io.BytesIO(raw)).convert("L")  # grayscale
        arr = np.asarray(img, dtype=np.float32)
    except Exception as e:  # noqa: BLE001 - surface any decode failure as 400
        raise HTTPException(status_code=400, detail=f"Could not read image: {e}")

    result = _get_predictor().predict(arr)
    return Prediction(**result.as_dict())


@app.post("/predict/canvas", response_model=Prediction)
def predict_canvas(req: CanvasRequest) -> Prediction:
    """Predict from a flat grayscale array drawn on the front-end canvas."""
    side = int(round(len(req.pixels) ** 0.5))
    arr = np.asarray(req.pixels, dtype=np.float32).reshape(side, side)
    result = _get_predictor().predict(arr, auto_orient=req.auto_orient)
    return Prediction(**result.as_dict())


# --------------------------------------------------------------------------- #
# Static UI
# --------------------------------------------------------------------------- #
@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse(UI_DIR / "index.html")


if UI_DIR.is_dir():
    app.mount("/static", StaticFiles(directory=UI_DIR), name="static")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("api.main:app", host="127.0.0.1", port=8000, reload=True)
