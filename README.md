# MNIST Digit Classification

Handwritten-digit classification on the classic MNIST dataset. This repo currently
contains the **exploratory data analysis (EDA) pipeline** that establishes a data
baseline before modeling.

## Layout

```
data/archive (1)/        # raw MNIST IDX files (tracked in git)
src/eda/
  ├── idx_loader.py      # robust IDX/ubyte loader (flat + nested layouts)
  └── run_eda.py         # full EDA pipeline → report + figures
eda_outputs/             # generated report + figures (gitignored)
requirements.txt
```

## Setup

```bash
pip install -r requirements.txt
```

Requires Python 3.9+ with `numpy`, `matplotlib`, `scikit-learn`.

## Data

The four standard MNIST IDX files are expected under `data/` (the loader tolerates
both the flat and the nested Kaggle-archive folder layout):

- `train-images.idx3-ubyte`, `train-labels.idx1-ubyte`
- `t10k-images.idx3-ubyte`, `t10k-labels.idx1-ubyte`

These binaries are committed to the repo, so the EDA runs out of the box after cloning.

## Run the EDA

From the repo root:

```bash
# Uses the bundled data/ dir by default; add --tsne for the non-linear cluster plot
python -m src.eda.run_eda --tsne

# Or point at a different data dir / output dir / seed
python -m src.eda.run_eda --data-dir "data/archive (1)" --out-dir eda_outputs --tsne --seed 0
```

This writes `eda_outputs/EDA_REPORT.md` and `eda_outputs/figures/*.png`. The run is
idempotent and only touches the output directory.

### What the pipeline covers

1. Integrity & schema — shapes, dtypes, value ranges, NaNs, label/image alignment
2. Class distribution & imbalance ratio (train + test)
3. Representative sample grid (10 per digit)
4. Per-class mean & std "average digit" images
5. Pixel-intensity profile & sparsity
6. Ink (foreground pixel count) per class
7. Exact-duplicate detection + train→test leakage check
8. Train vs. test mean-image drift
9. PCA: cumulative explained variance + 2-D scatter
10. t-SNE non-linear cluster view (optional, `--tsne`)
11. Furthest-from-centroid outliers per class (mislabel eyeballing)
12. Concrete modeling recommendations

## Reusing the loader in modeling code

```python
from src.eda import load_mnist

X_train, y_train, X_test, y_test = load_mnist("data/archive (1)")
```
