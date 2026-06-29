"""
mnist_classifier — modular MNIST handwritten-digit classification.

A Keras/TensorFlow fully-connected ANN (MLP) packaged for production use:

    mnist_classifier/
      config.py            centralized, override-from-env configuration
      data/loader.py       robust IDX loader + preprocessing
      models/mlp.py        Keras Sequential model factory
      training/train.py    end-to-end training pipeline
      training/evaluate.py retrain-free evaluation pipeline
      inference/predictor.py  load-once predictor for serving

The public surface most callers need:

    from mnist_classifier import Predictor, load_config
    predictor = Predictor.from_config(load_config())
    result = predictor.predict(image_28x28_uint8)
"""

from __future__ import annotations

from .config import Config, load_config

__all__ = ["Config", "load_config", "Predictor"]

__version__ = "1.0.0"


def __getattr__(name: str):
    # Lazy-import Predictor so that merely importing the package (e.g. for config)
    # does not pull in TensorFlow, which is slow to load.
    if name == "Predictor":
        from .inference.predictor import Predictor

        return Predictor
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
