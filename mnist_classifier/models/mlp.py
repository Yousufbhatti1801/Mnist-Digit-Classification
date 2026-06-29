"""
Keras Sequential MLP for MNIST digit classification.

Architecture (defaults): ``784 -> 256 -> 128 -> 10``, ReLU hidden activations,
dropout + L2 weight decay on every hidden layer, and a 10-way softmax head paired
with sparse categorical cross-entropy on integer labels.

Three independent regularizers (dropout, L2, and early stopping in the training
loop) keep a ~235k-parameter network from memorizing the 60k training images.
A dense MLP plateaus around ~98% test accuracy on MNIST; beating that needs a CNN.
"""

from __future__ import annotations

import os
from typing import Sequence

# Quiet TensorFlow's C++ logging before it is imported (0=all .. 3=errors only).
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

from tensorflow import keras
from tensorflow.keras import layers


def build_mlp(
    in_dim: int,
    hidden: Sequence[int],
    n_classes: int,
    dropout: float,
    weight_decay: float,
) -> keras.Model:
    """Build a regularized fully-connected classifier.

    Args:
        in_dim: flattened input size (784 for MNIST).
        hidden: hidden-layer widths, e.g. ``(256, 128)``.
        n_classes: number of output classes (10).
        dropout: dropout rate applied after each hidden layer.
        weight_decay: L2 kernel-regularization strength on hidden layers.

    Returns:
        An *uncompiled* ``keras.Sequential`` model. The training pipeline owns
        the optimizer/loss choice via ``model.compile``.
    """
    model = keras.Sequential(name="mnist_mlp")
    model.add(keras.Input(shape=(in_dim,)))
    for i, h in enumerate(hidden):
        model.add(
            layers.Dense(
                h,
                activation="relu",
                kernel_regularizer=keras.regularizers.l2(weight_decay),
                name=f"dense_{i}",
            )
        )
        model.add(layers.Dropout(dropout, name=f"dropout_{i}"))
    # Softmax head; paired with sparse categorical cross-entropy on integer labels.
    model.add(layers.Dense(n_classes, activation="softmax", name="probs"))
    return model
