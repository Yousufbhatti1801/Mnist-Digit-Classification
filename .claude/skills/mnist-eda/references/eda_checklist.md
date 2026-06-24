# MNIST EDA — rationale & interpretation guide

A reference for *why* each analysis exists and how to read its output. The
pipeline in `scripts/run_eda.py` implements all of these.

## 1. Integrity & schema
**Why:** Catch corrupt downloads, wrong dtype, off-by-one label ranges, and
image/label misalignment before they silently poison training.
**Expect for MNIST:** train `(60000,28,28)` uint8, test `(10000,28,28)`, labels
0–9, values 0–255, no NaNs.

## 2. Class distribution
**Why:** Imbalance changes the loss, the metric, and the validation split.
**Expect:** mild imbalance — ratio ≈ 1.24 (digit 1 most frequent, 5 least).
**Action:** ratio < ~1.3 → treat as balanced; use plain accuracy + a stratified
split. Larger ratios → consider class weights / macro-F1.

## 3. Sample grid
**Why:** Eyeball reality — confirm labels match images, see handwriting variety,
spot rotation/centering conventions.

## 4. Mean & std images
**Why:** The per-class mean is the "prototype" digit; the std map shows where
strokes vary most. Overlapping prototypes (4/9) preview confusable pairs.

## 5. Pixel-intensity profile
**Why:** Reveals sparsity and the bimodal (background-0 / ink-high) structure.
**Expect:** ~80% of pixels exactly 0. Justifies normalization and explains why
PCA compresses so well.

## 6. Ink per class
**Why:** Quantifies how much foreground each digit uses — a weak baseline feature
and a sanity check (1 is lightest, 0/8 heaviest).

## 7. Duplicates & leakage
**Why:** Duplicate train rows bias training; **test rows that appear in train
inflate the score**. This is the single most important integrity check.
**Expect for stock MNIST:** 0 duplicates, 0 leakage. Non-zero → investigate the
data source before trusting any reported accuracy.

## 8. Train/test drift
**Why:** If the test mean image differs materially from train, the test score
won't generalize to deployment. Compares per-pixel mean images.
**Expect:** near-identical means (global mean ≈ 33 for both).

## 9. PCA
**Why:** Measures linear redundancy and gives a fast 2-D separability view.
**Expect:** ~50 components capture 95% of variance (from 784). Strong case for
PCA preprocessing on classical models; partial class separation in 2-D.

## 10. t-SNE (optional)
**Why:** Non-linear view of cluster structure. Clean, separated clusters predict
that even simple models score well; overlaps (4/9, 3/5/8) predict confusions.
**Cost:** run on a PCA-reduced sample (~3k points) to keep it fast.

## 11. Outliers
**Why:** The sample furthest from its class centroid surfaces likely mislabels,
unusually-styled digits, or scanning artifacts — candidates for manual review or
robust-loss handling.

## 12. Modeling recommendations
The actionable hand-off. Normalize to [0,1]; flatten for classical ML / keep
28×28×1 for CNNs; stratified 10–15% validation split; baseline ladder
LogReg(~92%) → SVM-RBF(~98%) → CNN(99%+); mild augmentation for CNNs (no flips);
watch confusable pairs in the confusion matrix.

## Notes / gotchas
- **Never touch the 10k test set** until final evaluation — all EDA-driven
  decisions come from train (+ a held-out val split).
- The t-SNE/PCA scatters use a random subsample; fix `--seed` for reproducibility.
- The loader handles both flat and nested-folder IDX layouts and big-endian
  numeric IDX types, not just uint8.
