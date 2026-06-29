"""
Plotting and markdown-report helpers for the training and evaluation pipelines.

Kept separate from the pipelines so the numeric logic stays readable and the
matplotlib import is paid for only when a figure is actually drawn. matplotlib
is imported lazily inside each function for the same reason.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


def _plt():
    """Lazy, headless matplotlib (Agg backend, no display required)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #
def plot_curves(rows, best_epoch, path: Path) -> None:
    """Loss + accuracy learning curves, with the best (min val-loss) epoch marked."""
    plt = _plt()
    h = np.array(rows, dtype=float)
    ep, trl, tra, val, vaa = h[:, 0], h[:, 1], h[:, 2], h[:, 3], h[:, 4]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    ax1.plot(ep, trl, label="train"); ax1.plot(ep, val, label="val")
    ax1.axvline(best_epoch, ls="--", c="gray", lw=1, label=f"best (ep {best_epoch})")
    ax1.set_title("Loss"); ax1.set_xlabel("epoch"); ax1.set_ylabel("cross-entropy"); ax1.legend()
    ax2.plot(ep, tra, label="train"); ax2.plot(ep, vaa, label="val")
    ax2.axvline(best_epoch, ls="--", c="gray", lw=1)
    ax2.set_title("Accuracy"); ax2.set_xlabel("epoch"); ax2.set_ylabel("accuracy"); ax2.legend()
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)


def plot_confusion(cm: np.ndarray, path: Path) -> None:
    """Annotated test-set confusion matrix (counts)."""
    plt = _plt()
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


def plot_misclassified(X_raw, y_true, y_pred, path: Path, n: int = 25) -> None:
    """Grid of up to ``n`` misclassified test digits, captioned true->pred."""
    plt = _plt()
    wrong = np.where(y_pred != y_true)[0][:n]
    cols = 5
    rows = int(np.ceil(len(wrong) / cols)) if len(wrong) else 1
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.6, rows * 1.7))
    axes = np.atleast_1d(axes).ravel()
    for ax in axes:
        ax.axis("off")
    for ax, idx in zip(axes, wrong):
        ax.imshow(X_raw[idx], cmap="gray")
        ax.set_title(f"{y_true[idx]}->{y_pred[idx]}", fontsize=8, color="crimson")
    fig.suptitle(f"Misclassified test digits (first {len(wrong)})", fontsize=11)
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)


# --------------------------------------------------------------------------- #
# Markdown reports
# --------------------------------------------------------------------------- #
def write_training_report(path: Path, metrics: dict, rows, cm: np.ndarray) -> None:
    c = metrics["config"]
    overfit_gap = rows[-1][1] - rows[-1][3]  # train_loss - val_loss at last epoch
    per_class_acc = cm.diagonal() / cm.sum(1).clip(min=1)
    worst = np.argsort(per_class_acc)[:3]
    lines = [
        "# MNIST ANN — Training Report", "",
        f"Framework: {metrics['framework']}", "",
        "## Configuration",
        f"- Architecture: MLP `{c['in_dim']} -> {' -> '.join(map(str, c['hidden']))} -> {c['n_classes']}`",
        "- Activations: ReLU (hidden), softmax (output) with sparse categorical cross-entropy",
        f"- Regularization: dropout={c['dropout']}, L2 weight_decay={c['weight_decay']}, early stopping",
        f"- Optimizer: Adam (lr={c['lr']}), batch_size={c['batch_size']}",
        "- Preprocessing: flatten 28x28->784, normalize pixels /255 -> [0,1]",
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
        "- `best_model.keras` — full model (architecture + weights), reload with `keras.models.load_model`",
        "- `training_history.csv` — per-epoch curves",
        "- `metrics.json` — machine-readable summary",
        "- `figures/learning_curves.png`, `figures/confusion_matrix.png`", "",
        "## Next steps",
        "- If val loss kept dropping at the cap, raise `--epochs`.",
        "- If train acc >> val acc, increase `--dropout`/`--weight-decay` or shrink `--hidden`.",
        "- An MLP tops out ~98% on MNIST; for higher, move to a CNN (spatial structure).",
    ]
    Path(path).write_text("\n".join(lines) + "\n")


def write_evaluation_report(path: Path, evaluation: dict, report: dict, confusions, labels) -> None:
    acc = evaluation["accuracy"]
    macro = evaluation["macro_avg"]
    weighted = evaluation["weighted_avg"]
    worst = min(labels, key=lambda d: report[str(d)]["recall"])
    lines = [
        "# MNIST ANN — Evaluation Report", "",
        f"Held-out test-set evaluation of `{evaluation['model_path']}` "
        f"({evaluation['n_test']:,} images). Independent of training; re-run with "
        "`python -m scripts.evaluate`.", "",
        "## Headline",
        f"- **Test accuracy: {acc:.4f}** ({int(round(acc * evaluation['n_test']))} / {evaluation['n_test']:,} correct)",
        f"- Macro avg — precision {macro['precision']:.4f}, recall {macro['recall']:.4f}, F1 {macro['f1-score']:.4f}",
        f"- Weighted avg — precision {weighted['precision']:.4f}, recall {weighted['recall']:.4f}, F1 {weighted['f1-score']:.4f}",
        f"- Weakest digit by recall: **{worst}** (recall {report[str(worst)]['recall']:.4f})", "",
        "## Per-class metrics",
        "| digit | precision | recall | f1 | support |",
        "|---|---|---|---|---|",
    ]
    for d in labels:
        r = report[str(d)]
        lines.append(
            f"| {d} | {r['precision']:.4f} | {r['recall']:.4f} | "
            f"{r['f1-score']:.4f} | {int(r['support'])} |"
        )
    lines += ["", "## Top confusions (true -> predicted)"]
    for t, pr, ct in confusions:
        lines.append(f"- {t} -> {pr}: {ct} times")
    lines += [
        "", "## How to read this",
        "- **Recall** of digit *d* = of all true *d*s, how many were caught.",
        "- **Precision** of digit *d* = of everything called *d*, how many were right.",
        "- Confusions are off-diagonal cells of `figures/confusion_matrix.png`; "
        "`figures/misclassified.png` shows actual errors.", "",
        "## Artifacts",
        "- `evaluation.json` — machine-readable metrics",
        "- `figures/confusion_matrix.png`, `figures/misclassified.png`",
    ]
    Path(path).write_text("\n".join(lines) + "\n")
