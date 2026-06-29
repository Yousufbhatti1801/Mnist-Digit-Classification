"""
Training-time augmentation for the MNIST MLP.

A fully-connected MLP keys on absolute pixel positions, so it generalizes poorly
to hand-drawn digits that are shifted, slanted, or sized differently from MNIST.
Augmenting the *training* set with small random affine transforms (shift, rotate,
scale) teaches the model that a digit's identity is invariant to those changes,
which measurably improves real-world / canvas accuracy without changing the
architecture.

Augmentation is applied only to training data, freshly each epoch, via
``AugmentedSequence``. Validation and test data stay pristine so their metrics
remain comparable to the un-augmented baseline.
"""

from __future__ import annotations

import numpy as np

from .preprocess import normalize


def _affine_one(img: np.ndarray, angle_deg: float, scale: float,
                dx: float, dy: float) -> np.ndarray:
    """Apply rotation (deg), scale, and pixel shift to one HxW float image.

    Uses an inverse-map + bilinear sample so the output stays smooth (nearest
    sampling would alias thin strokes). Rotation/scale are about the image center.
    Dependency-free (pure numpy) to avoid pulling in scipy/cv2.
    """
    h, w = img.shape
    cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
    theta = np.deg2rad(angle_deg)
    cos, sin = np.cos(theta), np.sin(theta)

    ys, xs = np.mgrid[0:h, 0:w]
    # Output -> center-relative coords.
    yr = ys - cy - dy
    xr = xs - cx - dx
    # Inverse transform: undo scale then rotation to find the source pixel.
    yr /= scale
    xr /= scale
    src_y = (cos * yr + sin * xr) + cy
    src_x = (-sin * yr + cos * xr) + cx

    # Bilinear sample with zero padding outside the source image.
    x0 = np.floor(src_x).astype(int)
    y0 = np.floor(src_y).astype(int)
    x1, y1 = x0 + 1, y0 + 1
    wx = src_x - x0
    wy = src_y - y0

    def gather(yy, xx):
        valid = (yy >= 0) & (yy < h) & (xx >= 0) & (xx < w)
        out = np.zeros_like(img)
        out[valid] = img[np.clip(yy, 0, h - 1)[valid], np.clip(xx, 0, w - 1)[valid]]
        return out

    top = gather(y0, x0) * (1 - wx) + gather(y0, x1) * wx
    bot = gather(y1, x0) * (1 - wx) + gather(y1, x1) * wx
    return top * (1 - wy) + bot * wy


def augment_batch(images: np.ndarray, rng: np.random.Generator,
                  max_shift: float = 3.0, max_angle: float = 12.0,
                  scale_range: tuple[float, float] = (0.9, 1.1)) -> np.ndarray:
    """Randomly shift/rotate/scale each HxW image in a stack. Inputs/outputs [0,1]."""
    n = images.shape[0]
    angles = rng.uniform(-max_angle, max_angle, n)
    scales = rng.uniform(scale_range[0], scale_range[1], n)
    dxs = rng.uniform(-max_shift, max_shift, n)
    dys = rng.uniform(-max_shift, max_shift, n)
    out = np.empty_like(images)
    for i in range(n):
        out[i] = _affine_one(images[i], angles[i], scales[i], dxs[i], dys[i])
    return np.clip(out, 0.0, 1.0)


try:
    from tensorflow import keras

    class AugmentedSequence(keras.utils.Sequence):
        """Yields shuffled, freshly-augmented batches of flattened 784 vectors.

        ``X`` is the flat (N, 784) training matrix in [0, 1]; it is reshaped to
        28x28 internally, augmented, then re-flattened so the MLP sees the same
        784-vector shape it always has. A new random transform is drawn every
        epoch, so the model effectively never sees the same image twice.
        """

        def __init__(self, X: np.ndarray, y: np.ndarray, batch_size: int,
                     image_size: int = 28, seed: int = 0, **aug_kwargs):
            super().__init__()
            self.X = X.reshape(-1, image_size, image_size).astype(np.float32)
            self.y = y
            self.batch_size = batch_size
            self.image_size = image_size
            self.aug_kwargs = aug_kwargs
            self._rng = np.random.default_rng(seed)
            self._order = np.arange(len(y))
            self.on_epoch_end()

        def __len__(self) -> int:
            return int(np.ceil(len(self.y) / self.batch_size))

        def on_epoch_end(self) -> None:
            self._rng.shuffle(self._order)

        def __getitem__(self, i: int):
            idx = self._order[i * self.batch_size:(i + 1) * self.batch_size]
            imgs = augment_batch(self.X[idx], self._rng, **self.aug_kwargs)
            flat = imgs.reshape(len(idx), -1)
            return flat, self.y[idx]

except ImportError:  # keras not installed (e.g. config-only import) -> skip
    AugmentedSequence = None  # type: ignore
