"""
CLI entrypoint for evaluating a trained MNIST ANN on the test set.

Thin argparse layer over ``mnist_classifier.training.evaluate``. Loads the saved
model — no retraining — and writes the evaluation report, json, and figures.

Usage (from the repo root):
    python -m scripts.evaluate              # run as a module (preferred)
    python scripts/evaluate.py              # run as a file (also works)
    python -m scripts.evaluate --out-dir model_outputs --data-dir "data/archive (1)"
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running as a plain file (``python scripts/evaluate.py``) by putting the repo
# root on sys.path; running as a module (``python -m scripts.evaluate``) already does.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mnist_classifier.config import load_config
from mnist_classifier.training import evaluate


def main() -> None:
    p = argparse.ArgumentParser(description="Evaluate a trained MNIST ANN on the test set.")
    p.add_argument("--data-dir", help="Directory holding the MNIST IDX files.")
    p.add_argument("--out-dir", help="Where best_model.keras lives and outputs are written.")
    args = p.parse_args()

    config = load_config(**{k: v for k, v in vars(args).items() if v is not None})
    evaluate(config)


if __name__ == "__main__":
    main()
