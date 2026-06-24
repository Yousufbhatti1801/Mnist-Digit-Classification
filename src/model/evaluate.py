"""
Evaluate a trained MNIST ANN on the held-out test set.

Loads ``best_model.pt`` (weights + config) produced by ``train_ann.py`` and runs a
full classification evaluation on the official 10k MNIST test set — no retraining.
This is deliberately separate from training so model quality can be re-measured,
compared across runs, or audited without touching the training loop.

Usage (from the repo root):
    python -m src.model.evaluate                                  # bundled paths
    python -m src.model.evaluate --model-path model_outputs/best_model.pt
    python -m src.model.evaluate --data-dir DATA --out-dir model_outputs

Outputs (all under ``--out-dir``, nothing else is touched):
  - ``EVALUATION_REPORT.md``           accuracy, per-class precision/recall/F1,
                                       macro/weighted averages, top confusions
  - ``evaluation.json``                machine-readable metrics
  - ``figures/confusion_matrix.png``   test-set confusion matrix (counts)
  - ``figures/misclassified.png``      a grid of actual misclassified digits

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

# Works both as ``python -m src.model.evaluate`` and as a loose script.
try:
    from src.eda import load_mnist
    from src.model.train_ann import MLP, preprocess
except ImportError:  # repo root not on sys.path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from src.eda import load_mnist
    from src.model.train_ann import MLP, preprocess

import torch
from sklearn.metrics import classification_report, confusion_matrix


def _pick_device(requested: str) -> str:
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_model(model_path: str | Path, device: str):
    """Rebuild the MLP from a saved checkpoint and load its weights."""
    ckpt = torch.load(model_path, map_location=device)
    c = ckpt["config"]
    model = MLP(c["in_dim"], c["hidden"], c["n_classes"], c["dropout"]).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, c


def predict(model, X: np.ndarray, device: str, batch_size: int = 512) -> np.ndarray:
    """Batched argmax predictions for an N x 784 float32 array."""
    preds = []
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            xb = torch.from_numpy(X[i:i + batch_size]).to(device)
            preds.append(model(xb).argmax(1).cpu().numpy())
    return np.concatenate(preds)


def top_confusions(cm: np.ndarray, k: int = 8):
    """Return the k largest off-diagonal (true, pred, count) confusions."""
    pairs = []
    n = cm.shape[0]
    for t in range(n):
        for p in range(n):
            if t != p and cm[t, p] > 0:
                pairs.append((int(t), int(p), int(cm[t, p])))
    pairs.sort(key=lambda x: x[2], reverse=True)
    return pairs[:k]


def main():
    p = argparse.ArgumentParser(description="Evaluate a trained MNIST ANN on the test set.")
    p.add_argument("--model-path", default="model_outputs/best_model.pt",
                   help="Path to best_model.pt (weights + config).")
    p.add_argument("--data-dir", default="data/archive (1)",
                   help="Directory holding the MNIST IDX files.")
    p.add_argument("--out-dir", default="model_outputs",
                   help="Where to write the evaluation report, json, and figures.")
    p.add_argument("--device", default="auto", help="auto|cpu|cuda|mps")
    args = p.parse_args()

    device = _pick_device(args.device)
    print(f"[setup] device={device}")

    model_path = Path(args.model_path)
    if not model_path.is_file():
        raise FileNotFoundError(
            f"No model at {model_path}. Train one first: python -m src.model.train_ann"
        )

    out_dir = Path(args.out_dir)
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    # --- Load model + data -------------------------------------------------- #
    model, config = load_model(model_path, device)
    n_classes = config["n_classes"]
    _, _, X_test_raw, y_test = load_mnist(args.data_dir)
    X_test = preprocess(X_test_raw)
    print(f"[load] test {X_test_raw.shape}  model params="
          f"{sum(p.numel() for p in model.parameters()):,}")

    # --- Predict + metrics -------------------------------------------------- #
    preds = predict(model, X_test, device)
    accuracy = float((preds == y_test).mean())
    labels = list(range(n_classes))
    report = classification_report(
        y_test, preds, labels=labels, output_dict=True, zero_division=0,
    )
    cm = confusion_matrix(y_test, preds, labels=labels)
    confusions = top_confusions(cm)

    print(f"[result] test accuracy = {accuracy:.4f}  ({(preds == y_test).sum()} / {len(y_test)})")
    macro_f1 = report["macro avg"]["f1-score"]
    print(f"[result] macro F1 = {macro_f1:.4f}")
    print(f"[result] worst digit by recall: "
          f"{min(labels, key=lambda d: report[str(d)]['recall'])}")

    # --- Persist ------------------------------------------------------------ #
    evaluation = {
        "model_path": str(model_path),
        "n_test": int(len(y_test)),
        "accuracy": accuracy,
        "macro_avg": report["macro avg"],
        "weighted_avg": report["weighted avg"],
        "per_class": {str(d): report[str(d)] for d in labels},
        "top_confusions": [
            {"true": t, "pred": pr, "count": ct} for t, pr, ct in confusions
        ],
        "config": config,
    }
    with open(out_dir / "evaluation.json", "w") as f:
        json.dump(evaluation, f, indent=2)

    _plot_confusion(cm, fig_dir / "confusion_matrix.png")
    _plot_misclassified(X_test_raw, y_test, preds, fig_dir / "misclassified.png")
    _write_report(out_dir / "EVALUATION_REPORT.md", evaluation, report, confusions, labels)

    print(f"[done] wrote evaluation report, json, and figures to {out_dir}/")


# --------------------------------------------------------------------------- #
# Reporting helpers (matplotlib imported lazily)
# --------------------------------------------------------------------------- #
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


def _plot_misclassified(X_raw, y_true, y_pred, path, n=25):
    """Grid of up to n misclassified test digits, captioned true→pred."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    wrong = np.where(y_pred != y_true)[0][:n]
    cols = 5
    rows = int(np.ceil(len(wrong) / cols)) if len(wrong) else 1
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.6, rows * 1.7))
    axes = np.atleast_1d(axes).ravel()
    for ax in axes:
        ax.axis("off")
    for ax, idx in zip(axes, wrong):
        ax.imshow(X_raw[idx], cmap="gray")
        ax.set_title(f"{y_true[idx]}→{y_pred[idx]}", fontsize=8, color="crimson")
    fig.suptitle(f"Misclassified test digits (first {len(wrong)})", fontsize=11)
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)


def _write_report(path, evaluation, report, confusions, labels):
    acc = evaluation["accuracy"]
    macro = evaluation["macro_avg"]
    weighted = evaluation["weighted_avg"]
    worst = min(labels, key=lambda d: report[str(d)]["recall"])
    lines = [
        "# MNIST ANN — Evaluation Report", "",
        f"Held-out test-set evaluation of `{evaluation['model_path']}` "
        f"({evaluation['n_test']:,} images). Independent of training; re-run with "
        "`python -m src.model.evaluate`.", "",
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
    lines += [
        "", "## Top confusions (true → predicted)",
    ]
    for t, pr, ct in confusions:
        lines.append(f"- {t} → {pr}: {ct} times")
    lines += [
        "", "## How to read this",
        "- **Recall** of digit *d* = of all true *d*s, how many were caught. Low recall "
        "means the model misses that digit.",
        "- **Precision** of digit *d* = of everything called *d*, how many were right. "
        "Low precision means the model over-predicts that digit.",
        "- The confusions above are the off-diagonal cells of "
        "`figures/confusion_matrix.png`; `figures/misclassified.png` shows actual "
        "errors so you can judge whether they are genuinely ambiguous scrawls.", "",
        "## Artifacts",
        "- `evaluation.json` — machine-readable metrics",
        "- `figures/confusion_matrix.png`, `figures/misclassified.png`",
    ]
    Path(path).write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
