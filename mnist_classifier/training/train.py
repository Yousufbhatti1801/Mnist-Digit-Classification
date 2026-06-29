"""
End-to-end training pipeline for the MNIST ANN.

``train(config)`` runs the full pipeline and returns a metrics dict:
  1. Load raw IDX files.
  2. Preprocess (flatten + normalize).
  3. Stratified validation split.
  4. Build + compile the MLP.
  5. Fit with Adam + early stopping on val loss (best weights restored).
  6. Evaluate on val and the held-out test set; build a confusion matrix.
  7. Persist model, history CSV, metrics JSON, figures, and a markdown report.

Importable as a library (``from mnist_classifier.training import train``) and
driven by ``scripts/train.py`` for the CLI. Side-effect-free outside ``out_dir``.
"""

from __future__ import annotations

import json
import os

import numpy as np

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import tensorflow as tf
from tensorflow import keras

from ..config import Config
from ..data import AugmentedSequence, load_mnist, preprocess, stratified_val_split
from ..models import build_mlp
from . import reporting


def train(config: Config) -> dict:
    """Run the full training pipeline; return the metrics dict that is saved."""
    keras.utils.set_random_seed(config.seed)
    gpus = tf.config.list_physical_devices("GPU")
    print(f"[setup] tf={tf.__version__} keras={keras.__version__} seed={config.seed}")
    print(f"[setup] GPUs={[g.name for g in gpus] or 'none (CPU)'}")

    config.fig_dir.mkdir(parents=True, exist_ok=True)

    # --- Load + preprocess -------------------------------------------------- #
    X_train_raw, y_train, X_test_raw, y_test = load_mnist(config.data_dir)
    print(f"[load] train {X_train_raw.shape}  test {X_test_raw.shape}")
    X_train = preprocess(X_train_raw)
    X_test = preprocess(X_test_raw)
    y_train = y_train.astype(np.int64)
    y_test = y_test.astype(np.int64)

    tr_idx, va_idx = stratified_val_split(y_train, config.val_frac, config.seed)
    X_tr, y_tr = X_train[tr_idx], y_train[tr_idx]
    X_va, y_va = X_train[va_idx], y_train[va_idx]
    print(f"[split] train={len(tr_idx)}  val={len(va_idx)}  (stratified {config.val_frac:.0%})")

    # --- Build + compile ---------------------------------------------------- #
    model = build_mlp(
        config.in_dim, config.hidden, config.n_classes,
        config.dropout, config.weight_decay,
    )
    n_params = int(model.count_params())
    print(f"[model] MLP {config.in_dim}->{'->'.join(map(str, config.hidden))}->"
          f"{config.n_classes}  dropout={config.dropout}  l2={config.weight_decay}  "
          f"params={n_params:,}")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=config.lr),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    # --- Fit with early stopping on val loss -------------------------------- #
    early_stop = keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=config.patience,
        restore_best_weights=True, verbose=1,
    )
    if config.augment:
        # Augment training data on the fly (fresh transforms each epoch); keep
        # validation pristine so val_loss / early stopping stay comparable.
        print(f"[augment] on  shift=±{config.aug_max_shift}px  angle=±{config.aug_max_angle}°  "
              f"scale={config.aug_scale_min}-{config.aug_scale_max}")
        train_seq = AugmentedSequence(
            X_tr, y_tr, batch_size=config.batch_size,
            image_size=config.image_size, seed=config.seed,
            max_shift=config.aug_max_shift, max_angle=config.aug_max_angle,
            scale_range=(config.aug_scale_min, config.aug_scale_max),
        )
        history = model.fit(
            train_seq,
            validation_data=(X_va, y_va),
            epochs=config.epochs,
            callbacks=[early_stop],
            verbose=2,
        )
    else:
        print("[augment] off")
        history = model.fit(
            X_tr, y_tr,
            validation_data=(X_va, y_va),
            epochs=config.epochs,
            batch_size=config.batch_size,
            callbacks=[early_stop],
            verbose=2,
        )

    hist = history.history
    val_losses = hist["val_loss"]
    best_epoch = int(np.argmin(val_losses)) + 1  # 1-based; matches restored weights
    epochs_run = len(val_losses)
    print(f"[best] epoch={best_epoch}  val_loss={val_losses[best_epoch - 1]:.4f}")

    # --- Final evaluation (best weights already restored) ------------------- #
    val_loss, val_acc = model.evaluate(X_va, y_va, verbose=0)
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"[final] val  loss={val_loss:.4f} acc={val_acc:.4f}")
    print(f"[final] test loss={test_loss:.4f} acc={test_acc:.4f}")

    preds = model.predict(X_test, batch_size=512, verbose=0).argmax(1)
    cm = np.zeros((config.n_classes, config.n_classes), dtype=int)
    for t, pr in zip(y_test, preds):
        cm[t, pr] += 1

    # --- Persist ------------------------------------------------------------ #
    model.save(config.model_path)

    rows = list(zip(
        range(1, epochs_run + 1),
        hist["loss"], hist["accuracy"], hist["val_loss"], hist["val_accuracy"],
    ))
    with open(config.out_dir / "training_history.csv", "w") as f:
        f.write("epoch,train_loss,train_acc,val_loss,val_acc\n")
        for r in rows:
            f.write("{},{:.6f},{:.6f},{:.6f},{:.6f}\n".format(*r))

    metrics = {
        "best_epoch": best_epoch, "epochs_run": epochs_run,
        "val_loss": float(val_loss), "val_acc": float(val_acc),
        "test_loss": float(test_loss), "test_acc": float(test_acc),
        "n_params": n_params,
        "device": "GPU" if gpus else "CPU",
        "framework": f"tensorflow {tf.__version__} / keras {keras.__version__}",
        "config": config.as_dict(),
    }
    with open(config.out_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    reporting.plot_curves(rows, best_epoch, config.fig_dir / "learning_curves.png")
    reporting.plot_confusion(cm, config.fig_dir / "confusion_matrix.png")
    reporting.write_training_report(config.out_dir / "TRAINING_REPORT.md", metrics, rows, cm)

    print(f"[done] wrote model, metrics, figures, and report to {config.out_dir}/")
    return metrics
