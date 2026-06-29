"""
Centralized configuration for the MNIST classifier.

One ``Config`` dataclass holds every tunable knob — data location, model
hyperparameters, training schedule, and serving paths — so the training
pipeline, evaluation pipeline, and API all read from the same source of truth.

Defaults are sensible for the bundled MNIST data. Every field can be overridden
from the environment (prefix ``MNIST_``) so the same code runs unchanged in a
container or CI without editing source. CLI flags layer on top of this in the
``scripts/`` entrypoints.
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field
from pathlib import Path

# Repo root = two levels up from this file (mnist_classifier/config.py -> repo).
REPO_ROOT = Path(__file__).resolve().parents[1]


def _env(name: str, default: str) -> str:
    return os.environ.get(f"MNIST_{name}", default)


def _env_float(name: str, default: float) -> float:
    return float(os.environ.get(f"MNIST_{name}", default))


def _env_int(name: str, default: int) -> int:
    return int(os.environ.get(f"MNIST_{name}", default))


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(f"MNIST_{name}")
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


@dataclass
class Config:
    """All configuration for training, evaluation, and serving."""

    # --- Paths ------------------------------------------------------------- #
    data_dir: Path = field(default_factory=lambda: REPO_ROOT / _env("DATA_DIR", "data/archive (1)"))
    out_dir: Path = field(default_factory=lambda: REPO_ROOT / _env("OUT_DIR", "model_outputs"))

    # --- Model architecture ------------------------------------------------ #
    hidden: tuple[int, ...] = (256, 128)
    dropout: float = field(default_factory=lambda: _env_float("DROPOUT", 0.2))
    weight_decay: float = field(default_factory=lambda: _env_float("WEIGHT_DECAY", 1e-4))

    # --- Training schedule ------------------------------------------------- #
    lr: float = field(default_factory=lambda: _env_float("LR", 1e-3))
    batch_size: int = field(default_factory=lambda: _env_int("BATCH_SIZE", 128))
    epochs: int = field(default_factory=lambda: _env_int("EPOCHS", 30))
    patience: int = field(default_factory=lambda: _env_int("PATIENCE", 5))
    val_frac: float = field(default_factory=lambda: _env_float("VAL_FRAC", 0.1))
    seed: int = field(default_factory=lambda: _env_int("SEED", 0))

    # --- Augmentation (training only) -------------------------------------- #
    # Random shift/rotate/scale on the training set so the MLP generalizes to
    # hand-drawn digits that aren't framed exactly like MNIST. On by default.
    augment: bool = field(default_factory=lambda: _env_bool("AUGMENT", True))
    aug_max_shift: float = field(default_factory=lambda: _env_float("AUG_MAX_SHIFT", 3.0))
    aug_max_angle: float = field(default_factory=lambda: _env_float("AUG_MAX_ANGLE", 12.0))
    aug_scale_min: float = field(default_factory=lambda: _env_float("AUG_SCALE_MIN", 0.9))
    aug_scale_max: float = field(default_factory=lambda: _env_float("AUG_SCALE_MAX", 1.1))

    # --- Fixed dataset facts ----------------------------------------------- #
    image_size: int = 28          # MNIST images are 28x28
    n_classes: int = 10           # digits 0-9

    @property
    def in_dim(self) -> int:
        """Flattened input dimension (784 for MNIST)."""
        return self.image_size * self.image_size

    @property
    def model_path(self) -> Path:
        """Where the trained Keras model is saved / loaded from."""
        return self.out_dir / "best_model.keras"

    @property
    def fig_dir(self) -> Path:
        return self.out_dir / "figures"

    def as_dict(self) -> dict:
        """JSON-serializable view (Paths -> str, tuple -> list).

        Includes the computed ``in_dim`` so consumers (reports, metrics.json) get
        a self-contained record of the architecture without recomputing it.
        """
        d = asdict(self)
        d["data_dir"] = str(self.data_dir)
        d["out_dir"] = str(self.out_dir)
        d["hidden"] = list(self.hidden)
        d["in_dim"] = self.in_dim  # property, not a field -> add explicitly
        return d


def load_config(**overrides) -> Config:
    """Build a ``Config``, applying keyword overrides on top of env/defaults.

    ``hidden`` accepts a comma-separated string ("512,256,128") or a sequence,
    so CLI flags can pass it through verbatim.
    """
    if "hidden" in overrides and isinstance(overrides["hidden"], str):
        overrides["hidden"] = tuple(
            int(h) for h in overrides["hidden"].split(",") if h.strip()
        )
    cfg = Config()
    for k, v in overrides.items():
        if v is None:
            continue
        if k in ("data_dir", "out_dir"):
            v = Path(v)
        setattr(cfg, k, v)
    return cfg
