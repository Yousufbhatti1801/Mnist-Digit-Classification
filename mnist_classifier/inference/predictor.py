"""
Load-once predictor for serving the trained MNIST model.

A ``Predictor`` loads ``best_model.keras`` exactly once and exposes ``predict``
for a single image and ``predict_batch`` for many. It owns the messy job of
turning *arbitrary* user input — an uploaded PNG of any size, a canvas drawing,
a raw 28x28 array — into the exact 784-vector the model was trained on:

  * grayscale + resize to 28x28,
  * orient to MNIST's convention (white digit on black background),
  * flatten + normalize to [0, 1].

Keeping this beside the model (not in the API) means the same preprocessing is
reused by any caller — API, batch script, or notebook — with no train/serve skew.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

from ..config import Config, load_config


@dataclass
class PredictionResult:
    """One prediction: the winning digit, its confidence, and the full distribution."""

    digit: int
    confidence: float
    probabilities: list[float]  # length 10, index == digit

    def as_dict(self) -> dict:
        return {
            "digit": self.digit,
            "confidence": self.confidence,
            "probabilities": self.probabilities,
        }


class Predictor:
    """Wraps a loaded Keras model + the exact preprocessing it expects."""

    def __init__(self, model, image_size: int = 28):
        self._model = model
        self._image_size = image_size

    # --- construction ------------------------------------------------------ #
    @classmethod
    def from_config(cls, config: Config | None = None) -> "Predictor":
        config = config or load_config()
        return cls.from_path(config.model_path, config.image_size)

    @classmethod
    def from_path(cls, model_path: str | Path, image_size: int = 28) -> "Predictor":
        model_path = Path(model_path)
        if not model_path.is_file():
            raise FileNotFoundError(
                f"No model at {model_path}. Train one first: python -m scripts.train"
            )
        from tensorflow import keras  # imported here so import-time stays cheap

        return cls(keras.models.load_model(model_path), image_size)

    # --- input normalization ----------------------------------------------- #
    def _to_mnist_vector(self, image: np.ndarray, auto_orient: bool = True) -> np.ndarray:
        """Coerce an arbitrary grayscale image into a 784-vector in [0, 1].

        Accepts a 28x28 array (used directly) or any HxW / HxWxC array. The latter
        is treated as a *real-world drawing* (canvas / uploaded photo) and put
        through the same spatial normalization the MNIST authors applied: the
        digit is cropped to its ink, scaled so its longest side fits a 20px box,
        and re-centered by center-of-mass in the 28x28 frame. Without this, an
        off-center or differently-sized stroke looks nothing like the training
        data and the model — 98% on MNIST — guesses badly on hand-drawn input.

        If ``auto_orient`` and the image looks like dark-ink-on-light-paper
        (mean > 0.5 once in [0, 1]), it is inverted to MNIST's white-on-black.

        Returns a (1, 784) float32 array already scaled to [0, 1] — i.e. fully
        preprocessed. We normalize *here* and flatten by hand rather than calling
        ``preprocess`` (which would normalize a second time).
        """
        arr = np.asarray(image, dtype=np.float32)
        if arr.ndim == 3:  # drop channels -> luminance
            arr = arr.mean(axis=2)
        if arr.max() > 1.0:  # came in as 0-255
            arr = arr / 255.0

        already_mnist = arr.shape == (self._image_size, self._image_size)
        if auto_orient and arr.mean() > 0.5:
            arr = 1.0 - arr

        if already_mnist:
            # Pre-formatted MNIST array (e.g. a test-set image) -> use verbatim.
            return arr.reshape(1, -1)

        # Real-world drawing: orient first (so "ink" is the bright foreground),
        # then crop / scale / recenter the way MNIST was built.
        return _mnist_center(arr, self._image_size).reshape(1, -1)

    # --- prediction -------------------------------------------------------- #
    def predict(self, image: np.ndarray, auto_orient: bool = True) -> PredictionResult:
        """Predict the digit in a single image."""
        vec = self._to_mnist_vector(image, auto_orient)
        probs = self._model.predict(vec, verbose=0)[0]
        digit = int(probs.argmax())
        return PredictionResult(
            digit=digit,
            confidence=float(probs[digit]),
            probabilities=[float(p) for p in probs],
        )

    def predict_batch(self, images, auto_orient: bool = True) -> list[PredictionResult]:
        """Predict over many images (each preprocessed independently)."""
        vecs = np.vstack([self._to_mnist_vector(im, auto_orient) for im in images])
        probs = self._model.predict(vecs, verbose=0)
        out = []
        for p in probs:
            d = int(p.argmax())
            out.append(PredictionResult(d, float(p[d]), [float(x) for x in p]))
        return out


def _mnist_center(arr: np.ndarray, size: int) -> np.ndarray:
    """Replicate MNIST's spatial normalization on a white-on-black [0,1] image.

    The original MNIST pipeline: crop the digit to its bounding box, scale it so
    the longest side is 20px (preserving aspect ratio), then place it in a 28x28
    field centered on the digit's center of mass. Reproducing this at inference
    time removes train/serve skew for hand-drawn / uploaded digits.

    Falls back to a plain resize if the image is effectively blank.
    """
    ink = arr > 0.10  # foreground mask; threshold tolerates anti-aliased edges
    if not ink.any():
        return _resize(arr, size)  # nothing drawn -> best effort

    rows, cols = np.where(ink)
    r0, r1 = rows.min(), rows.max() + 1
    c0, c1 = cols.min(), cols.max() + 1
    crop = arr[r0:r1, c0:c1]

    # Scale longest side to 20px (the MNIST inner box), keeping aspect ratio.
    box = 20
    h, w = crop.shape
    if h >= w:
        new_h, new_w = box, max(1, int(round(w * box / h)))
    else:
        new_h, new_w = max(1, int(round(h * box / w))), box
    scaled = _resize(crop, (new_h, new_w))

    # Center by center of mass within the 28x28 field.
    canvas = np.zeros((size, size), dtype=np.float32)
    total = scaled.sum()
    if total > 0:
        cy = (np.arange(new_h)[:, None] * scaled).sum() / total
        cx = (np.arange(new_w)[None, :] * scaled).sum() / total
    else:
        cy, cx = new_h / 2.0, new_w / 2.0
    top = int(round(size / 2.0 - cy))
    left = int(round(size / 2.0 - cx))
    top = max(0, min(size - new_h, top))
    left = max(0, min(size - new_w, left))
    canvas[top:top + new_h, left:left + new_w] = scaled
    return canvas


def _resize(arr: np.ndarray, size) -> np.ndarray:
    """Resize a 2-D float array, preferring Pillow, else nearest-neighbor.

    ``size`` is an int for a square target or a ``(height, width)`` tuple.
    """
    out_h, out_w = (size, size) if isinstance(size, int) else size
    try:
        from PIL import Image

        lo, hi = float(arr.min()), float(arr.max())
        scaled = (arr - lo) / (hi - lo) * 255.0 if hi > lo else np.zeros_like(arr)
        # PIL .resize takes (width, height).
        img = Image.fromarray(scaled.astype(np.uint8)).resize((out_w, out_h), Image.BILINEAR)
        return np.asarray(img, dtype=np.float32) / 255.0 * (hi - lo) + lo
    except ImportError:
        # Dependency-free fallback: integer nearest-neighbor sampling.
        h, w = arr.shape
        ys = (np.linspace(0, h - 1, out_h)).round().astype(int)
        xs = (np.linspace(0, w - 1, out_w)).round().astype(int)
        return arr[np.ix_(ys, xs)]
