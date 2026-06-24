"""MNIST exploratory data analysis package."""

from .idx_loader import find_mnist_files, load_mnist, read_idx

__all__ = ["load_mnist", "read_idx", "find_mnist_files"]
