"""
Fast, dependency-light tests for the data, config, and inference layers.

These exercise the pieces that have nothing to do with a long training run, so
they finish in seconds. Run with:  python -m pytest tests/  (or just python this
file). The API and a trained model are tested in test_api.py when a model exists.
"""

from __future__ import annotations

import numpy as np

from mnist_classifier.config import load_config
from mnist_classifier.data import load_mnist, preprocess, stratified_val_split


def test_config_overrides_and_paths():
    cfg = load_config(epochs=7, hidden="512,128", dropout=0.5)
    assert cfg.epochs == 7
    assert cfg.hidden == (512, 128)
    assert cfg.dropout == 0.5
    assert cfg.in_dim == 784
    assert cfg.model_path.name == "best_model.keras"


def test_preprocess_shape_and_range():
    img = np.random.randint(0, 256, size=(5, 28, 28), dtype=np.uint8)
    x = preprocess(img)
    assert x.shape == (5, 784)
    assert x.dtype == np.float32
    assert 0.0 <= x.min() and x.max() <= 1.0
    # single image -> (1, 784)
    assert preprocess(img[0]).shape == (1, 784)


def test_stratified_split_is_proportional():
    y = np.repeat(np.arange(10), 100)  # 100 of each class
    tr, va = stratified_val_split(y, val_frac=0.1, seed=0)
    assert len(va) == 100 and len(tr) == 900
    # every class contributes exactly 10 to val
    assert np.bincount(y[va], minlength=10).tolist() == [10] * 10
    # no overlap between train and val indices
    assert set(tr).isdisjoint(set(va))


def test_loader_shapes():
    cfg = load_config()
    Xtr, ytr, Xte, yte = load_mnist(cfg.data_dir)
    assert Xtr.shape == (60000, 28, 28) and ytr.shape == (60000,)
    assert Xte.shape == (10000, 28, 28) and yte.shape == (10000,)
    assert set(np.unique(ytr).tolist()) == set(range(10))


if __name__ == "__main__":
    test_config_overrides_and_paths()
    test_preprocess_shape_and_range()
    test_stratified_split_is_proportional()
    test_loader_shapes()
    print("all pipeline tests passed")
