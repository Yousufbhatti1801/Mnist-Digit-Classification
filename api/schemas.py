"""
Pydantic request/response schemas for the MNIST prediction API.

Keeping the wire contract in one place gives FastAPI automatic validation and a
self-documenting OpenAPI schema at ``/docs``.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, field_validator


class CanvasRequest(BaseModel):
    """A digit drawn on the front-end canvas, sent as a flat grayscale array.

    ``pixels`` is row-major: either 784 values (already 28x28) or any N*N square
    that the server will downsample. Values may be 0-1 or 0-255; the predictor
    normalizes and auto-orients, so the client need not match MNIST's convention.
    """

    pixels: list[float] = Field(..., min_length=1, description="Row-major grayscale pixels.")
    auto_orient: bool = Field(True, description="Invert dark-on-light input to MNIST white-on-black.")

    @field_validator("pixels")
    @classmethod
    def _must_be_square(cls, v: list[float]) -> list[float]:
        n = len(v)
        side = int(round(n ** 0.5))
        if side * side != n:
            raise ValueError(f"pixels length {n} is not a perfect square (got side ~{side}).")
        return v


class Prediction(BaseModel):
    """The model's verdict for one image."""

    digit: int = Field(..., ge=0, le=9, description="Predicted digit 0-9.")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Softmax probability of the winner.")
    probabilities: list[float] = Field(..., description="Full 10-way softmax distribution, index == digit.")


class HealthResponse(BaseModel):
    # ``model_*`` field names are fine here; opt out of Pydantic's protected namespace.
    model_config = ConfigDict(protected_namespaces=())

    status: str
    model_loaded: bool
    model_path: str
    n_classes: int
