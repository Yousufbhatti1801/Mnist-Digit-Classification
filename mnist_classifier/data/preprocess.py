"""
Preprocessing for the MNIST ANN.

The model is a fully-connected MLP, so every image must be flattened to a 784
vector and pixels scaled to [0, 1]. Keeping this in one place guarantees the
*exact same* transform is applied at train time and at inference time — the most
common source of train/serve skew.
"""

from __future__ import annotations

import numpy as np


def normalize(images: np.ndarray) -> np.ndarray:
    """Scale uint8 pixels [0, 255] -> float32 [0, 1] without flattening."""
    return images.astype(np.float32) / 255.0


def preprocess(images: np.ndarray) -> np.ndarray:
    """Flatten N x 28 x 28 -> N x 784 and scale pixels to [0, 1] float32.

    Also accepts a single 28x28 image (returns 1 x 784) so inference code can
    pass one image without reshaping first.
    """
    if images.ndim == 2:  # a single H x W image
        images = images[None, ...]
    flat = images.reshape(images.shape[0], -1)
    return normalize(flat)


def stratified_val_split(y: np.ndarray, val_frac: float, seed: int):
    """Return (train_idx, val_idx) holding out ``val_frac`` of *each* class.

    Stratifying keeps every digit proportionally represented in validation, so
    the val metric is not biased by a class the random split happened to starve.
    """
    rng = np.random.default_rng(seed)
    train_idx, val_idx = [], []
    for cls in np.unique(y):
        idx = np.where(y == cls)[0]
        rng.shuffle(idx)
        n_val = int(round(len(idx) * val_frac))
        val_idx.append(idx[:n_val])
        train_idx.append(idx[n_val:])
    train_idx = np.concatenate(train_idx)
    val_idx = np.concatenate(val_idx)
    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    return train_idx, val_idx
