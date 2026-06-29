"""
Retrain-free evaluation of a saved MNIST model on the held-out test set.

``evaluate(config)`` loads ``best_model.keras`` and runs a full classification
evaluation on the official 10k test set: accuracy, per-class precision/recall/F1,
macro/weighted averages, the largest confusions, and a gallery of actual
misclassified digits. Independent of training so quality can be re-measured or
compared across runs without touching the training loop.
"""

from __future__ import annotations

import json
import os

import numpy as np

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

from tensorflow import keras
from sklearn.metrics import classification_report, confusion_matrix

from ..config import Config
from ..data import load_mnist, preprocess
from . import reporting


def top_confusions(cm: np.ndarray, k: int = 8):
    """Return the ``k`` largest off-diagonal (true, pred, count) confusions."""
    pairs = []
    n = cm.shape[0]
    for t in range(n):
        for p in range(n):
            if t != p and cm[t, p] > 0:
                pairs.append((int(t), int(p), int(cm[t, p])))
    pairs.sort(key=lambda x: x[2], reverse=True)
    return pairs[:k]


def evaluate(config: Config) -> dict:
    """Evaluate the saved model on the test set; return the evaluation dict."""
    if not config.model_path.is_file():
        raise FileNotFoundError(
            f"No model at {config.model_path}. Train one first: python -m scripts.train"
        )
    config.fig_dir.mkdir(parents=True, exist_ok=True)

    model = keras.models.load_model(config.model_path)
    _, _, X_test_raw, y_test = load_mnist(config.data_dir)
    y_test = y_test.astype(np.int64)
    X_test = preprocess(X_test_raw)
    print(f"[load] test {X_test_raw.shape}  model params={model.count_params():,}")

    preds = model.predict(X_test, batch_size=512, verbose=0).argmax(1)
    accuracy = float((preds == y_test).mean())
    labels = list(range(config.n_classes))
    report = classification_report(
        y_test, preds, labels=labels, output_dict=True, zero_division=0,
    )
    cm = confusion_matrix(y_test, preds, labels=labels)
    confusions = top_confusions(cm)

    print(f"[result] test accuracy = {accuracy:.4f}  "
          f"({(preds == y_test).sum()} / {len(y_test)})")
    print(f"[result] macro F1 = {report['macro avg']['f1-score']:.4f}")

    evaluation = {
        "model_path": str(config.model_path),
        "n_test": int(len(y_test)),
        "accuracy": accuracy,
        "macro_avg": report["macro avg"],
        "weighted_avg": report["weighted avg"],
        "per_class": {str(d): report[str(d)] for d in labels},
        "top_confusions": [
            {"true": t, "pred": pr, "count": ct} for t, pr, ct in confusions
        ],
        "config": config.as_dict(),
    }
    with open(config.out_dir / "evaluation.json", "w") as f:
        json.dump(evaluation, f, indent=2)

    reporting.plot_confusion(cm, config.fig_dir / "confusion_matrix.png")
    reporting.plot_misclassified(X_test_raw, y_test, preds, config.fig_dir / "misclassified.png")
    reporting.write_evaluation_report(
        config.out_dir / "EVALUATION_REPORT.md", evaluation, report, confusions, labels
    )

    print(f"[done] wrote evaluation report, json, and figures to {config.out_dir}/")
    return evaluation
