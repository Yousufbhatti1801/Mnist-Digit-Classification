"""
MNIST exploratory data analysis pipeline.

Loads the IDX files, runs a sequence of senior-DS-grade analyses, writes every
figure to ``<out>/figures/`` and a human-readable ``<out>/EDA_REPORT.md`` that
ends with concrete modeling recommendations.

Usage (from the repo root):
    python -m src.eda.run_eda                       # uses the bundled data/ dir
    python src/eda/run_eda.py --tsne                # add the non-linear plot
    python src/eda/run_eda.py --data-dir DATA --out-dir eda_outputs --seed 0

Designed to be re-runnable and side-effect-free outside the output directory.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")  # FAISS/OpenMP safety on macOS

import numpy as np

import matplotlib

matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt

# Allow both ``python -m src.eda.run_eda`` and ``python src/eda/run_eda.py``.
try:
    from .idx_loader import load_mnist
except ImportError:  # run as a loose script
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from idx_loader import load_mnist

CLASSES = list(range(10))

# Repo root is two levels up from this file (src/eda/run_eda.py -> repo).
REPO_ROOT = Path(__file__).resolve().parents[2]
# The dataset ships inside a Kaggle-style archive folder; the loader tolerates
# the flat-vs-nested mess, so pointing at the archive root is enough.
DEFAULT_DATA_DIR = REPO_ROOT / "data" / "archive (1)"
DEFAULT_OUT_DIR = REPO_ROOT / "eda_outputs"


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _save(fig, figdir: Path, name: str) -> str:
    path = figdir / name
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return f"figures/{name}"


def _section(lines: list[str], title: str) -> None:
    lines.append(f"\n## {title}\n")


# --------------------------------------------------------------------------- #
# analyses — each appends markdown to `report` and may write figures
# --------------------------------------------------------------------------- #
def integrity(report, Xtr, ytr, Xte, yte):
    _section(report, "1. Dataset integrity & schema")
    rows = [
        ("train images", Xtr.shape, Xtr.dtype, int(Xtr.min()), int(Xtr.max())),
        ("test images", Xte.shape, Xte.dtype, int(Xte.min()), int(Xte.max())),
    ]
    report.append("| split | shape | dtype | min | max |")
    report.append("|---|---|---|---|---|")
    for name, shape, dt, lo, hi in rows:
        report.append(f"| {name} | {shape} | {dt} | {lo} | {hi} |")
    report.append("")
    report.append(f"- Train labels: {ytr.shape}, range [{ytr.min()}, {ytr.max()}]")
    report.append(f"- Test labels: {yte.shape}, range [{yte.min()}, {yte.max()}]")
    nan_tr = np.isnan(Xtr).any() if np.issubdtype(Xtr.dtype, np.floating) else False
    report.append(f"- NaNs present: {bool(nan_tr)}")
    label_mismatch = (set(np.unique(ytr)) | set(np.unique(yte))) - set(CLASSES)
    report.append(f"- Unexpected label values: {label_mismatch or 'none'}")
    report.append(f"- Images-per-label aligned: "
                  f"{Xtr.shape[0] == ytr.shape[0]} (train), "
                  f"{Xte.shape[0] == yte.shape[0]} (test)")


def class_distribution(report, figdir, ytr, yte):
    _section(report, "2. Class distribution & balance")
    ctr = np.bincount(ytr, minlength=10)
    cte = np.bincount(yte, minlength=10)
    report.append("| digit | train | train % | test | test % |")
    report.append("|---|---|---|---|---|")
    for d in CLASSES:
        report.append(
            f"| {d} | {ctr[d]} | {100*ctr[d]/ctr.sum():.2f} | "
            f"{cte[d]} | {100*cte[d]/cte.sum():.2f} |"
        )
    imb = ctr.max() / ctr.min()
    report.append("")
    report.append(f"- Train imbalance ratio (max/min): **{imb:.3f}** "
                  f"({'balanced' if imb < 1.2 else 'mild imbalance'}).")

    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].bar(CLASSES, ctr, color="#4C72B0")
    ax[0].set_title("Train class counts"); ax[0].set_xticks(CLASSES)
    ax[1].bar(CLASSES, cte, color="#DD8452")
    ax[1].set_title("Test class counts"); ax[1].set_xticks(CLASSES)
    report.append(f"\n![class distribution]({_save(fig, figdir, 'class_distribution.png')})")


def sample_grid(report, figdir, X, y, rng):
    _section(report, "3. Representative samples per class")
    fig, axes = plt.subplots(10, 10, figsize=(10, 10))
    for d in CLASSES:
        idx = np.where(y == d)[0]
        pick = rng.choice(idx, 10, replace=False)
        for j, p in enumerate(pick):
            ax = axes[d, j]
            ax.imshow(X[p], cmap="gray"); ax.axis("off")
            if j == 0:
                ax.set_ylabel(str(d), rotation=0, labelpad=12, fontsize=12)
    fig.suptitle("10 random samples per digit (rows 0-9)")
    report.append(f"![samples]({_save(fig, figdir, 'sample_grid.png')})")


def mean_std_images(report, figdir, X, y):
    _section(report, "4. Average & variability digit ('eigen-ink')")
    Xf = X.astype(np.float32)
    fig, axes = plt.subplots(2, 10, figsize=(13, 3))
    for d in CLASSES:
        cls = Xf[y == d]
        axes[0, d].imshow(cls.mean(0), cmap="gray"); axes[0, d].axis("off")
        axes[0, d].set_title(str(d))
        axes[1, d].imshow(cls.std(0), cmap="magma"); axes[1, d].axis("off")
    axes[0, 0].set_ylabel("mean"); axes[1, 0].set_ylabel("std")
    fig.suptitle("Per-class mean (top) and std (bottom) images")
    report.append(f"![mean/std]({_save(fig, figdir, 'mean_std_images.png')})")
    report.append("\n- Bright std regions show where each digit varies most "
                  "(stroke style); useful for understanding confusable pairs.")


def pixel_intensity(report, figdir, X):
    _section(report, "5. Pixel-intensity profile")
    flat = X.reshape(-1)
    frac_zero = float((flat == 0).mean())
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].hist(flat, bins=50, color="#55A868")
    ax[0].set_title("All pixels"); ax[0].set_yscale("log")
    ax[0].set_xlabel("intensity 0-255"); ax[0].set_ylabel("count (log)")
    nz = flat[flat > 0]
    ax[1].hist(nz, bins=50, color="#C44E52")
    ax[1].set_title("Non-zero (ink) pixels"); ax[1].set_xlabel("intensity 1-255")
    report.append(f"![pixel intensity]({_save(fig, figdir, 'pixel_intensity.png')})")
    report.append(f"\n- **{100*frac_zero:.1f}%** of all pixels are exactly 0 "
                  "(black background) — the data is sparse.")
    report.append(f"- Mean intensity: {flat.mean():.2f}; "
                  f"mean ink intensity: {nz.mean():.2f}.")


def ink_per_class(report, figdir, X, y):
    _section(report, "6. Ink quantity per class")
    ink = (X > 0).reshape(X.shape[0], -1).sum(1)  # nonzero pixel count per image
    data = [ink[y == d] for d in CLASSES]
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.boxplot(data, tick_labels=CLASSES, showfliers=False)
    ax.set_xlabel("digit"); ax.set_ylabel("# ink pixels")
    ax.set_title("Ink (foreground pixel count) distribution per class")
    report.append(f"![ink]({_save(fig, figdir, 'ink_per_class.png')})")
    means = {d: float(ink[y == d].mean()) for d in CLASSES}
    lo = min(means, key=means.get); hi = max(means, key=means.get)
    report.append(f"\n- Lightest digit: **{lo}** ({means[lo]:.0f} px); "
                  f"heaviest: **{hi}** ({means[hi]:.0f} px). "
                  "Ink count alone is a weak but nonzero signal.")


def duplicates(report, Xtr, Xte):
    _section(report, "7. Duplicate detection")
    def hashes(X):
        return [hashlib.md5(row.tobytes()).hexdigest() for row in X]
    htr = hashes(Xtr)
    n_dup_tr = len(htr) - len(set(htr))
    set_tr = set(htr)
    leak = sum(1 for h in hashes(Xte) if h in set_tr)
    report.append(f"- Exact duplicate images within train: **{n_dup_tr}**")
    report.append(f"- Test images that also appear in train (leakage): **{leak}**")
    if leak:
        report.append("  - ⚠️ Investigate before trusting the test score.")


def train_test_drift(report, figdir, Xtr, Xte):
    _section(report, "8. Train vs. test distribution (drift)")
    mtr = Xtr.astype(np.float32).mean(0)
    mte = Xte.astype(np.float32).mean(0)
    diff = mte - mtr
    fig, ax = plt.subplots(1, 3, figsize=(11, 3.5))
    ax[0].imshow(mtr, cmap="gray"); ax[0].set_title("train mean"); ax[0].axis("off")
    ax[1].imshow(mte, cmap="gray"); ax[1].set_title("test mean"); ax[1].axis("off")
    im = ax[2].imshow(diff, cmap="bwr"); ax[2].set_title("test - train"); ax[2].axis("off")
    fig.colorbar(im, ax=ax[2], fraction=0.046)
    report.append(f"![drift]({_save(fig, figdir, 'train_test_drift.png')})")
    report.append(f"\n- Max abs per-pixel mean difference: {np.abs(diff).max():.2f} "
                  f"(on 0-255 scale). Global mean train={mtr.mean():.2f}, "
                  f"test={mte.mean():.2f}.")


def pca_view(report, figdir, X, y, rng, n=4000):
    _section(report, "9. PCA — variance & 2-D class separability")
    from sklearn.decomposition import PCA

    idx = rng.choice(X.shape[0], min(n, X.shape[0]), replace=False)
    Xs = X[idx].reshape(len(idx), -1).astype(np.float32) / 255.0
    ys = y[idx]
    full = PCA(n_components=50, random_state=0).fit(Xs)
    cum = np.cumsum(full.explained_variance_ratio_)
    k95 = int(np.searchsorted(cum, 0.95) + 1)
    proj = full.transform(Xs)[:, :2]

    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    ax[0].plot(range(1, 51), cum, marker=".")
    ax[0].axhline(0.95, ls="--", c="gray")
    ax[0].set_title("Cumulative explained variance")
    ax[0].set_xlabel("# components"); ax[0].set_ylabel("cum. variance")
    sc = ax[1].scatter(proj[:, 0], proj[:, 1], c=ys, cmap="tab10", s=6, alpha=0.6)
    ax[1].set_title(f"PC1 vs PC2 ({len(idx)} samples)")
    fig.colorbar(sc, ax=ax[1], ticks=CLASSES)
    report.append(f"![pca]({_save(fig, figdir, 'pca.png')})")
    report.append(f"\n- **{k95} principal components** capture 95% of variance "
                  f"(down from 784 raw dims) — strong redundancy, PCA/whitening viable.")


def tsne_view(report, figdir, X, y, rng, n=3000):
    _section(report, "10. t-SNE — non-linear cluster structure")
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE

    idx = rng.choice(X.shape[0], min(n, X.shape[0]), replace=False)
    Xs = X[idx].reshape(len(idx), -1).astype(np.float32) / 255.0
    ys = y[idx]
    Xp = PCA(n_components=50, random_state=0).fit_transform(Xs)  # denoise first
    emb = TSNE(n_components=2, init="pca", perplexity=30, random_state=0).fit_transform(Xp)
    fig, ax = plt.subplots(figsize=(7, 6))
    sc = ax.scatter(emb[:, 0], emb[:, 1], c=ys, cmap="tab10", s=7, alpha=0.7)
    fig.colorbar(sc, ax=ax, ticks=CLASSES)
    ax.set_title(f"t-SNE of {len(idx)} samples"); ax.axis("off")
    report.append(f"![tsne]({_save(fig, figdir, 'tsne.png')})")
    report.append("\n- Well-separated clusters ⇒ even simple classifiers should do well; "
                  "overlapping clusters (e.g. 4/9, 3/5/8) flag likely confusions.")


def outliers(report, figdir, X, y):
    _section(report, "11. Atypical samples (possible mislabels / hard cases)")
    Xf = X.reshape(X.shape[0], -1).astype(np.float32)
    far_idx, far_d = [], []
    for d in CLASSES:
        mask = np.where(y == d)[0]
        centroid = Xf[mask].mean(0)
        dist = np.linalg.norm(Xf[mask] - centroid, axis=1)
        far_idx.append(mask[dist.argmax()]); far_d.append(d)
    fig, axes = plt.subplots(1, 10, figsize=(13, 1.8))
    for j, (p, d) in enumerate(zip(far_idx, far_d)):
        axes[j].imshow(X[p], cmap="gray"); axes[j].axis("off")
        axes[j].set_title(f"'{d}'", fontsize=9)
    fig.suptitle("Furthest-from-centroid sample per class — eyeball for mislabels")
    report.append(f"![outliers]({_save(fig, figdir, 'outliers.png')})")


def recommendations(report, Xtr):
    _section(report, "12. Modeling recommendations")
    report.append(
        "- **Normalize** to [0,1] (`/255.0`) or standardize per-pixel; "
        "raw 0-255 hurts gradient-based training.\n"
        "- **Reshape**: flatten to 784-vectors for classical ML "
        "(LogReg/SVM/RandomForest); keep 28×28×1 for CNNs.\n"
        "- **Validation**: carve a stratified 10–15% split from the 60k train set; "
        "keep the 10k test set untouched until final evaluation.\n"
        "- **Baselines** (expected accuracy): LogReg ~92%, SVM(RBF) ~98%, "
        "simple CNN ~99%+. Use these as sanity checkpoints.\n"
        "- **Augmentation** worth trying for CNNs: small rotations/shifts/zoom "
        "(±10°, ±2px). Avoid flips — they corrupt digit identity.\n"
        "- **Watch confusables**: 4↔9, 3↔5↔8, 7↔1 — inspect these in the "
        "confusion matrix after the first model.\n"
        "- **Dimensionality reduction** (PCA→~50 dims) is a cheap, strong "
        "preprocessing step for classical models."
    )


# --------------------------------------------------------------------------- #
# driver
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description="MNIST EDA pipeline")
    ap.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR),
                    help="dir holding the MNIST IDX files (default: bundled data/)")
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    ap.add_argument("--tsne", action="store_true", help="run t-SNE (slower)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    out = Path(args.out_dir)
    figdir = out / "figures"
    figdir.mkdir(parents=True, exist_ok=True)

    print("Loading MNIST...")
    Xtr, ytr, Xte, yte = load_mnist(args.data_dir)

    report: list[str] = [
        "# MNIST — Exploratory Data Analysis",
        f"\n_Generated {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')} "
        f"from `{args.data_dir}` (seed={args.seed})._",
    ]

    integrity(report, Xtr, ytr, Xte, yte)
    class_distribution(report, figdir, ytr, yte)
    sample_grid(report, figdir, Xtr, ytr, rng)
    mean_std_images(report, figdir, Xtr, ytr)
    pixel_intensity(report, figdir, Xtr)
    ink_per_class(report, figdir, Xtr, ytr)
    duplicates(report, Xtr, Xte)
    train_test_drift(report, figdir, Xtr, Xte)
    pca_view(report, figdir, Xtr, ytr, rng)
    if args.tsne:
        tsne_view(report, figdir, Xtr, ytr, rng)
    outliers(report, figdir, Xtr, ytr)
    recommendations(report, Xtr)

    report_path = out / "EDA_REPORT.md"
    report_path.write_text("\n".join(report) + "\n")
    print(f"Done. Report: {report_path}  |  Figures: {figdir}/")


if __name__ == "__main__":
    main()
