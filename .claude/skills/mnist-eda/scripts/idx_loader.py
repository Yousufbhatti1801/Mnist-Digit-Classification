"""
Robust loader for MNIST-style IDX (ubyte) files.

The IDX format header is:
  bytes 0-1 : 0x0000 (magic prefix)
  byte  2   : data type code (0x08 = unsigned byte, the only type MNIST uses)
  byte  3   : number of dimensions (1 for labels, 3 for images)
  next 4*ndim bytes : each dimension size, big-endian uint32
  remainder : the payload, row-major

This module also handles the messy real-world layout where the same archive
ships the files both flat (``train-images.idx3-ubyte``) and nested inside a
same-named folder (``train-images-idx3-ubyte/train-images-idx3-ubyte``).
"""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np

# Data-type codes defined by the IDX spec -> numpy dtypes.
_IDX_DTYPES = {
    0x08: np.uint8,
    0x09: np.int8,
    0x0B: np.dtype(">i2"),
    0x0C: np.dtype(">i4"),
    0x0D: np.dtype(">f4"),
    0x0E: np.dtype(">f8"),
}

# Canonical MNIST roles -> the filename stems we might see on disk.
_ROLE_PATTERNS = {
    "train_images": ["train-images.idx3-ubyte", "train-images-idx3-ubyte"],
    "train_labels": ["train-labels.idx1-ubyte", "train-labels-idx1-ubyte"],
    "test_images": ["t10k-images.idx3-ubyte", "t10k-images-idx3-ubyte"],
    "test_labels": ["t10k-labels.idx1-ubyte", "t10k-labels-idx1-ubyte"],
}


def read_idx(path: str | Path) -> np.ndarray:
    """Read a single IDX file into a numpy array (images returned as N x H x W)."""
    path = Path(path)
    with open(path, "rb") as f:
        zero1, zero2, type_code, ndim = struct.unpack(">BBBB", f.read(4))
        if zero1 != 0 or zero2 != 0:
            raise ValueError(f"{path}: bad IDX magic prefix {zero1},{zero2}")
        if type_code not in _IDX_DTYPES:
            raise ValueError(f"{path}: unknown IDX data-type code {type_code:#x}")
        shape = struct.unpack(f">{ndim}I", f.read(4 * ndim))
        dtype = _IDX_DTYPES[type_code]
        data = np.frombuffer(f.read(), dtype=dtype)
    expected = int(np.prod(shape)) if shape else 0
    if data.size != expected:
        raise ValueError(
            f"{path}: payload size {data.size} != product of shape {shape} ({expected})"
        )
    # Cast big-endian numeric types to native for downstream math.
    if dtype != np.uint8:
        data = data.astype(np.dtype(dtype).newbyteorder("="))
    return data.reshape(shape)


def _resolve(data_dir: Path, candidates: list[str]) -> Path | None:
    """Find a file for a role, tolerating flat or nested same-named-folder layouts."""
    for name in candidates:
        flat = data_dir / name
        if flat.is_file():
            return flat
        nested = data_dir / name / name
        if nested.is_file():
            return nested
    # Last resort: recursive glob on a distinctive token.
    token = candidates[0].split(".")[0]
    hits = sorted(p for p in data_dir.rglob("*") if p.is_file() and token in p.name)
    return hits[0] if hits else None


def find_mnist_files(data_dir: str | Path) -> dict[str, Path]:
    """Locate the four canonical MNIST files under ``data_dir``."""
    data_dir = Path(data_dir)
    resolved: dict[str, Path] = {}
    missing = []
    for role, candidates in _ROLE_PATTERNS.items():
        hit = _resolve(data_dir, candidates)
        if hit is None:
            missing.append(role)
        else:
            resolved[role] = hit
    if missing:
        raise FileNotFoundError(
            f"Could not locate {missing} under {data_dir}. Found: "
            + ", ".join(str(p.relative_to(data_dir)) for p in resolved.values())
        )
    return resolved


def load_mnist(data_dir: str | Path):
    """Return (X_train, y_train, X_test, y_test) as numpy arrays."""
    files = find_mnist_files(data_dir)
    return (
        read_idx(files["train_images"]),
        read_idx(files["train_labels"]),
        read_idx(files["test_images"]),
        read_idx(files["test_labels"]),
    )


if __name__ == "__main__":
    import sys

    d = sys.argv[1] if len(sys.argv) > 1 else "."
    Xtr, ytr, Xte, yte = load_mnist(d)
    print(f"train images {Xtr.shape} {Xtr.dtype}  labels {ytr.shape}")
    print(f"test  images {Xte.shape} {Xte.dtype}  labels {yte.shape}")
