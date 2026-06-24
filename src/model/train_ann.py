"""
Train a fully-connected ANN (multi-layer perceptron) on MNIST.

Pipeline, in order:
  1. Load the raw IDX files via ``src.eda.load_mnist`` (the repo's shared loader).
  2. Preprocess: flatten each 28x28 image to a 784 vector and normalize pixels
     from [0, 255] -> [0, 1] (float32).
  3. Carve a stratified validation split out of the training set so every digit
     is represented proportionally.
  4. Build an MLP whose width/depth is sized for MNIST and regularized (ReLU +
     dropout + weight decay) so it does not overfit.
  5. Train with Adam, tracking train/val loss and accuracy each epoch, and
     early-stop on the validation loss (restoring the best weights).
  6. Evaluate the restored best model on the held-out test set.

Usage (from the repo root):
    python -m src.model.train_ann                          # bundled data/ dir
    python -m src.model.train_ann --epochs 50 --dropout 0.3
    python -m src.model.train_ann --data-dir DATA --out-dir model_outputs

Outputs (all under ``--out-dir``, nothing else is touched):
  - ``best_model.pt``        best-epoch weights + the config needed to rebuild
  - ``training_history.csv`` per-epoch train/val loss & accuracy
  - ``metrics.json``         final val/test loss & accuracy, best epoch, config
  - ``figures/learning_curves.png``   loss + accuracy curves (val-loss min marked)
  - ``figures/confusion_matrix.png``  test-set confusion matrix
  - ``TRAINING_REPORT.md``   human-readable summary + interpretation

Design rationale lives in ``.claude/skills/mnist-ann/references/ann_design_notes.md``.
Designed to be re-runnable and side-effect-free outside the output directory.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")  # FAISS/OpenMP safety on macOS

import numpy as np

# Reuse the repo's shared IDX loader. Works both as ``python -m src.model.train_ann``
# and as a loose script run directly.
try:
    from src.eda import load_mnist
except ImportError:  # run as a loose script, repo root not on sys.path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from src.eda import load_mnist

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


# --------------------------------------------------------------------------- #
# Preprocessing
# --------------------------------------------------------------------------- #
def preprocess(images: np.ndarray) -> np.ndarray:
    """Flatten N x 28 x 28 -> N x 784 and scale pixels to [0, 1] float32."""
    flat = images.reshape(images.shape[0], -1).astype(np.float32)
    return flat / 255.0


def stratified_val_split(y: np.ndarray, val_frac: float, seed: int):
    """Return (train_idx, val_idx) holding out ``val_frac`` of *each* class."""
    rng = np.random.default_rng(seed)
    train_idx, val_idx = [], []
    for cls in np.unique(y):
        idx = np.where(y == cls)[0]
        rng.shuffle(idx)
        n_val = int(round(len(idx) * val_frac))
        val_idx.append(idx[:n_val])
        train_idx.append(idx[n_val:])
    train_idx = np.concatenate(train_idx)
    val_idx = np.concatenate(val_idx)
    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    return train_idx, val_idx


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #
class MLP(nn.Module):
    """A small, regularized MLP: [in -> (Linear+ReLU+Dropout) x H -> out]."""

    def __init__(self, in_dim: int, hidden, n_classes: int, dropout: float):
        super().__init__()
        layers = []
        prev = in_dim
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ReLU(inplace=True), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, n_classes))  # logits; CE loss applies softmax
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


# --------------------------------------------------------------------------- #
# Train / eval loops
# --------------------------------------------------------------------------- #
def _run_epoch(model, loader, criterion, device, optimizer=None):
    """One pass over ``loader``. Trains if ``optimizer`` is given, else evals."""
    train = optimizer is not None
    model.train(train)
    total_loss, total_correct, total = 0.0, 0, 0
    torch.set_grad_enabled(train)
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        logits = model(xb)
        loss = criterion(logits, yb)
        if train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        total_loss += loss.item() * xb.size(0)
        total_correct += (logits.argmax(1) == yb).sum().item()
        total += xb.size(0)
    torch.set_grad_enabled(True)
    return total_loss / total, total_correct / total


def main():
    p = argparse.ArgumentParser(description="Train an ANN on MNIST.")
    p.add_argument("--data-dir", default="data/archive (1)",
                   help="Directory holding the MNIST IDX files.")
    p.add_argument("--out-dir", default="model_outputs",
                   help="Where to write model, metrics, figures, report.")
    p.add_argument("--hidden", default="256,128",
                   help="Comma-separated hidden-layer widths (default 256,128).")
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--epochs", type=int, default=30, help="Max epochs (early stop usually fires first).")
    p.add_argument("--patience", type=int, default=5, help="Early-stop patience on val loss.")
    p.add_argument("--val-frac", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="auto", help="auto|cpu|cuda|mps")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if args.device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    else:
        device = args.device
    print(f"[setup] device={device} seed={args.seed}")

    out_dir = Path(args.out_dir)
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    # --- Load + preprocess -------------------------------------------------- #
    X_train_raw, y_train, X_test_raw, y_test = load_mnist(args.data_dir)
    print(f"[load] train {X_train_raw.shape}  test {X_test_raw.shape}")
    X_train = preprocess(X_train_raw)
    X_test = preprocess(X_test_raw)
    in_dim = X_train.shape[1]
    n_classes = int(y_train.max()) + 1

    tr_idx, va_idx = stratified_val_split(y_train, args.val_frac, args.seed)
    print(f"[split] train={len(tr_idx)}  val={len(va_idx)}  (stratified {args.val_frac:.0%})")

    def to_loader(X, y, idx=None, shuffle=False):
        if idx is not None:
            X, y = X[idx], y[idx]
        ds = TensorDataset(torch.from_numpy(X), torch.from_numpy(y.astype(np.int64)))
        return DataLoader(ds, batch_size=args.batch_size, shuffle=shuffle)

    train_loader = to_loader(X_train, y_train, tr_idx, shuffle=True)
    val_loader = to_loader(X_train, y_train, va_idx)
    test_loader = to_loader(X_test, y_test)

    # --- Build model -------------------------------------------------------- #
    hidden = [int(h) for h in str(args.hidden).split(",") if h.strip()]
    model = MLP(in_dim, hidden, n_classes, args.dropout).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[model] MLP {in_dim}->{'->'.join(map(str, hidden))}->{n_classes}  "
          f"dropout={args.dropout}  params={n_params:,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
                                 weight_decay=args.weight_decay)

    # --- Train with early stopping on val loss ------------------------------ #
    history = []
    best_val_loss = float("inf")
    best_epoch = 0
    best_state = None
    epochs_no_improve = 0

    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc = _run_epoch(model, train_loader, criterion, device, optimizer)
        va_loss, va_acc = _run_epoch(model, val_loader, criterion, device)
        history.append((epoch, tr_loss, tr_acc, va_loss, va_acc))
        flag = ""
        if va_loss < best_val_loss - 1e-4:
            best_val_loss, best_epoch = va_loss, epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
            flag = "  *best*"
        else:
            epochs_no_improve += 1
        print(f"[epoch {epoch:2d}] train_loss={tr_loss:.4f} acc={tr_acc:.4f} | "
              f"val_loss={va_loss:.4f} acc={va_acc:.4f}{flag}")
        if epochs_no_improve >= args.patience:
            print(f"[early-stop] no val-loss improvement for {args.patience} epochs.")
            break

    # Restore best weights before final evaluation.
    if best_state is not None:
        model.load_state_dict(best_state)
    print(f"[best] epoch={best_epoch}  val_loss={best_val_loss:.4f}")

    # --- Final evaluation --------------------------------------------------- #
    val_loss, val_acc = _run_epoch(model, val_loader, criterion, device)
    test_loss, test_acc = _run_epoch(model, test_loader, criterion, device)
    print(f"[final] val  loss={val_loss:.4f} acc={val_acc:.4f}")
    print(f"[final] test loss={test_loss:.4f} acc={test_acc:.4f}")

    # Confusion matrix on test.
    model.eval()
    preds = []
    with torch.no_grad():
        for xb, _ in test_loader:
            preds.append(model(xb.to(device)).argmax(1).cpu().numpy())
    preds = np.concatenate(preds)
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for t, pr in zip(y_test, preds):
        cm[t, pr] += 1

    # --- Persist outputs ---------------------------------------------------- #
    config = {
        "in_dim": in_dim, "hidden": hidden, "n_classes": n_classes,
        "dropout": args.dropout, "lr": args.lr, "weight_decay": args.weight_decay,
        "batch_size": args.batch_size, "max_epochs": args.epochs,
        "patience": args.patience, "val_frac": args.val_frac, "seed": args.seed,
    }
    torch.save({"state_dict": model.state_dict(), "config": config}, out_dir / "best_model.pt")

    with open(out_dir / "training_history.csv", "w") as f:
        f.write("epoch,train_loss,train_acc,val_loss,val_acc\n")
        for row in history:
            f.write("{},{:.6f},{:.6f},{:.6f},{:.6f}\n".format(*row))

    metrics = {
        "best_epoch": best_epoch, "epochs_run": history[-1][0],
        "val_loss": val_loss, "val_acc": val_acc,
        "test_loss": test_loss, "test_acc": test_acc,
        "n_params": n_params, "device": device, "config": config,
    }
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    _plot_curves(history, best_epoch, fig_dir / "learning_curves.png")
    _plot_confusion(cm, fig_dir / "confusion_matrix.png")
    _write_report(out_dir / "TRAINING_REPORT.md", metrics, history, cm)

    print(f"[done] wrote model, metrics, figures, and report to {out_dir}/")


# --------------------------------------------------------------------------- #
# Reporting helpers (matplotlib imported lazily so --help stays fast)
# --------------------------------------------------------------------------- #
def _plot_curves(history, best_epoch, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    h = np.array(history)
    ep, trl, tra, val, vaa = h[:, 0], h[:, 1], h[:, 2], h[:, 3], h[:, 4]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    ax1.plot(ep, trl, label="train"); ax1.plot(ep, val, label="val")
    ax1.axvline(best_epoch, ls="--", c="gray", lw=1, label=f"best (ep {best_epoch})")
    ax1.set_title("Loss"); ax1.set_xlabel("epoch"); ax1.set_ylabel("cross-entropy"); ax1.legend()
    ax2.plot(ep, tra, label="train"); ax2.plot(ep, vaa, label="val")
    ax2.axvline(best_epoch, ls="--", c="gray", lw=1)
    ax2.set_title("Accuracy"); ax2.set_xlabel("epoch"); ax2.set_ylabel("accuracy"); ax2.legend()
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)


def _plot_confusion(cm, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_title("Test confusion matrix")
    ax.set_xlabel("predicted"); ax.set_ylabel("true")
    ax.set_xticks(range(cm.shape[0])); ax.set_yticks(range(cm.shape[0]))
    thresh = cm.max() / 2
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black", fontsize=7)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)


def _write_report(path, metrics, history, cm):
    c = metrics["config"]
    overfit_gap = history[-1][1] - history[-1][3]  # train_loss - val_loss at last epoch
    per_class_acc = cm.diagonal() / cm.sum(1).clip(min=1)
    worst = np.argsort(per_class_acc)[:3]
    lines = [
        "# MNIST ANN — Training Report", "",
        "## Configuration",
        f"- Architecture: MLP `{c['in_dim']} -> {' -> '.join(map(str, c['hidden']))} -> {c['n_classes']}`",
        "- Activations: ReLU (hidden), softmax via cross-entropy loss (output)",
        f"- Regularization: dropout={c['dropout']}, weight_decay={c['weight_decay']}, early stopping",
        f"- Optimizer: Adam (lr={c['lr']}), batch_size={c['batch_size']}",
        f"- Preprocessing: flatten 28x28->784, normalize pixels /255 -> [0,1]",
        f"- Stratified validation split: {c['val_frac']:.0%}",
        f"- Trainable parameters: {metrics['n_params']:,}  |  device: {metrics['device']}", "",
        "## Results",
        f"- Best epoch (min val loss): **{metrics['best_epoch']}** of {metrics['epochs_run']} run",
        f"- Validation: loss **{metrics['val_loss']:.4f}**, accuracy **{metrics['val_acc']:.4f}**",
        f"- Test: loss **{metrics['test_loss']:.4f}**, accuracy **{metrics['test_acc']:.4f}**", "",
        "## Overfitting check",
        f"- Final-epoch train vs val loss gap: {overfit_gap:+.4f} "
        f"({'healthy' if overfit_gap > -0.05 else 'val worse than train — watch for overfit'})",
        "- Early stopping restored the best-val-loss weights, so the saved model is the "
        "generalization optimum, not the last (possibly overfit) epoch.", "",
        "## Hardest digits (lowest per-class test accuracy)",
    ]
    for cls in worst:
        lines.append(f"- digit {cls}: {per_class_acc[cls]:.4f}")
    lines += [
        "", "## Artifacts",
        "- `best_model.pt` — best weights + config (reload to rebuild the exact model)",
        "- `training_history.csv` — per-epoch curves",
        "- `metrics.json` — machine-readable summary",
        "- `figures/learning_curves.png`, `figures/confusion_matrix.png`", "",
        "## Next steps",
        "- If val loss kept dropping at the cap, raise `--epochs`.",
        "- If train acc >> val acc, increase `--dropout`/`--weight-decay` or shrink `--hidden`.",
        "- An MLP tops out ~98% on MNIST; for higher, move to a CNN (spatial structure).",
    ]
    Path(path).write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
