"""
CLI entrypoint for training the MNIST ANN.

Thin argparse layer over ``mnist_classifier.training.train``: every flag maps to a
``Config`` field, so the same defaults serve the library, the API, and the CLI.

Usage (from the repo root):
    python -m scripts.train                 # run as a module (preferred)
    python scripts/train.py                 # run as a file (also works)
    python -m scripts.train --hidden 512,256,128 --dropout 0.3 --epochs 50
    python -m scripts.train --data-dir "data/archive (1)" --out-dir model_outputs
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running as a plain file (``python scripts/train.py``) by putting the repo
# root on sys.path; running as a module (``python -m scripts.train``) already does.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mnist_classifier.config import load_config
from mnist_classifier.training import train


def main() -> None:
    p = argparse.ArgumentParser(description="Train an ANN on MNIST (Keras/TensorFlow).")
    p.add_argument("--data-dir", help="Directory holding the MNIST IDX files.")
    p.add_argument("--out-dir", help="Where to write model, metrics, figures, report.")
    p.add_argument("--hidden", help="Comma-separated hidden-layer widths (default 256,128).")
    p.add_argument("--dropout", type=float)
    p.add_argument("--lr", type=float)
    p.add_argument("--weight-decay", type=float, dest="weight_decay",
                   help="L2 kernel-regularization strength.")
    p.add_argument("--batch-size", type=int, dest="batch_size")
    p.add_argument("--epochs", type=int, help="Max epochs (early stop usually fires first).")
    p.add_argument("--patience", type=int, help="Early-stop patience on val loss.")
    p.add_argument("--val-frac", type=float, dest="val_frac")
    p.add_argument("--seed", type=int)
    aug = p.add_mutually_exclusive_group()
    aug.add_argument("--augment", dest="augment", action="store_true", default=None,
                     help="Enable train-time shift/rotate/scale augmentation (default on).")
    aug.add_argument("--no-augment", dest="augment", action="store_false",
                     help="Disable augmentation (train on raw MNIST only).")
    p.add_argument("--aug-max-shift", type=float, dest="aug_max_shift")
    p.add_argument("--aug-max-angle", type=float, dest="aug_max_angle")
    args = p.parse_args()

    config = load_config(**{k: v for k, v in vars(args).items() if v is not None})
    train(config)


if __name__ == "__main__":
    main()
