---
name: mnist-eda
description: Run a senior-data-scientist EDA on the MNIST (or any IDX/ubyte) digit-classification dataset — loads the raw IDX files and produces figures plus a markdown report covering integrity, class balance, pixel/ink profiles, duplicates & leakage, train/test drift, PCA/t-SNE separability, outliers, and modeling recommendations. Use when starting work on the MNIST dataset, when asked to explore/profile/analyze digit-image data in IDX format, or to establish a data baseline before modeling.
---

# MNIST EDA

## When to use
Use at the start of an MNIST digit-classification project (or any dataset shipped
as MNIST-style IDX/ubyte files) to understand the data before modeling. It answers:
is the data clean and balanced, how separable are the classes, is there train/test
leakage or drift, and what preprocessing will the model need.

## Prerequisites
- Python 3 with `numpy`, `matplotlib`, `scikit-learn` (and they should already be
  present on this machine). If missing: `pip install -r scripts/requirements.txt`.
- The four standard MNIST files somewhere under a data directory, in either layout:
  - flat: `train-images.idx3-ubyte`, `train-labels.idx1-ubyte`, `t10k-images.idx3-ubyte`, `t10k-labels.idx1-ubyte`
  - or nested in same-named folders (the loader auto-resolves both).

## Instructions

### Step 1: Locate the data
Find the directory holding the IDX files. For this project it is `data/`. Confirm
with a quick listing if unsure. The loader tolerates the flat-vs-nested-folder mess
common in Kaggle archives.

### Step 2: Run the pipeline
From the project root:

```bash
python3 ~/.claude/skills/mnist-eda/scripts/run_eda.py \
    --data-dir data \
    --out-dir eda_outputs \
    --tsne          # optional, adds the non-linear cluster plot (~30-60s)
```

This writes `eda_outputs/EDA_REPORT.md` and `eda_outputs/figures/*.png`. The run is
idempotent and only touches the output directory. Use `--seed` for reproducible
sampling; drop `--tsne` for a faster run.

### Step 3: Sanity-check the load
The script prints the loaded shapes first. Expect `(60000, 28, 28)` train /
`(10000, 28, 28)` test for standard MNIST. If shapes look wrong, the data directory
or file format is the problem — fix that before reading further.

### Step 4: Read the report and summarize for the user
Open `eda_outputs/EDA_REPORT.md`. Walk the user through the 12 sections, leading
with anything actionable: class imbalance, any duplicates/leakage (section 7),
train/test drift (section 8), and the modeling recommendations (section 12).
Reference specific figures by path so the user can open them.

### Step 5: Hand off to modeling
Close by restating the concrete next steps from section 12 (normalize, stratified
val split, baseline ladder LogReg→SVM→CNN, confusable pairs to watch). This is the
bridge that lets the project "proceed further."

## What the pipeline covers
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
`scripts/idx_loader.py` is standalone and reusable:

```python
from idx_loader import load_mnist
X_train, y_train, X_test, y_test = load_mnist("data")
```

## Extending
- Add an analysis: write a `def my_analysis(report, figdir, ...)` that appends
  markdown to `report` and saves figures via the `_save` helper, then call it from
  `main()`. Keep each analysis self-contained and side-effect-free outside `figdir`.
- See `references/eda_checklist.md` for the rationale behind each step and how to
  interpret the figures.
