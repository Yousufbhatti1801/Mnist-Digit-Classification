"""Training and evaluation pipelines."""

from __future__ import annotations

from .evaluate import evaluate
from .train import train

__all__ = ["train", "evaluate"]
