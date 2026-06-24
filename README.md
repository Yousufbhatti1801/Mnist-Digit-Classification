# MNIST Digit Classification

Handwritten-digit classification on the classic MNIST dataset. This repo contains
two pipelines: an **exploratory data analysis (EDA) pipeline** that establishes a
data baseline, and a **model-training pipeline** that trains a regularized
fully-connected ANN (MLP) end-to-end.

## Layout

```
data/archive (1)/        # raw MNIST IDX files (tracked in git)
src/eda/
  ├── idx_loader.py      # robust IDX/ubyte loader (flat + nested layouts)
  └── run_eda.py         # full EDA pipeline → report + figures
src/model/
  └── train_ann.py       # ANN/MLP training pipeline → model + metrics + figures
eda_outputs/             # generated EDA report + figures (gitignored)
model_outputs/           # generated model, metrics, figures, report (gitignored)
requirements.txt
```

## Setup

```bash
pip install -r requirements.txt
```

Requires Python 3.9+ with `numpy`, `matplotlib`, `scikit-learn`, and `torch`.

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

## Train the ANN

From the repo root (run the EDA first to sanity-check the data):

```bash
# Sensible defaults: MLP 784→256→128→10, ReLU + dropout 0.2 + weight decay 1e-4,
# Adam lr 1e-3, batch 128, up to 30 epochs with early stopping on val loss.
python -m src.model.train_ann

# Override any hyperparameter
python -m src.model.train_ann --hidden 512,256,128 --dropout 0.3 --epochs 50
python -m src.model.train_ann --data-dir "data/archive (1)" --out-dir model_outputs --device cpu
```

This writes `model_outputs/TRAINING_REPORT.md`, `best_model.pt`, `metrics.json`,
`training_history.csv`, and `figures/{learning_curves,confusion_matrix}.png`. The
run is idempotent and only touches the output directory.

### What the pipeline does

1. **Preprocess** — flatten 28×28→784, normalize pixels `/255` → [0, 1] float32
2. **Stratified 10% validation split** — every digit represented proportionally;
   the official 10k test set is reserved for one final unbiased pass
3. **Build** — regularized MLP: `[Linear→ReLU→Dropout] × N → 10-way logit head`
4. **Train** — Adam + cross-entropy, tracking train/val loss & accuracy per epoch
5. **Regularize against overfit** — dropout + weight decay + early stopping on val
   loss with best-weight restore (three independent defenses)
6. **Evaluate** — restored best model on val and test; test confusion matrix

A dense MLP plateaus around **~98% test accuracy** (the current run reaches 98.2%);
a CNN is the next step past ~99%. Design rationale for every hyperparameter lives in
`.claude/skills/mnist-ann/references/ann_design_notes.md`.

### Evaluate the model

Training already reports test accuracy + a confusion matrix. For a deeper, retrain-free
evaluation of the saved model (per-class precision/recall/F1, macro/weighted averages,
ranked confusions, and a gallery of actual misclassified digits):

```bash
python -m src.model.evaluate
python -m src.model.evaluate --model-path model_outputs/best_model.pt --data-dir "data/archive (1)"
```

This writes `model_outputs/EVALUATION_REPORT.md`, `evaluation.json`, and
`figures/{confusion_matrix,misclassified}.png`. It loads `best_model.pt` and runs only
the test set — no training — so model quality can be re-measured or compared across runs.

### Reloading the trained model

```python
import torch
from src.model import MLP, preprocess

ckpt = torch.load("model_outputs/best_model.pt")
c = ckpt["config"]
model = MLP(c["in_dim"], c["hidden"], c["n_classes"], c["dropout"])
model.load_state_dict(ckpt["state_dict"]); model.eval()
# X = preprocess(raw_images_N_28_28); logits = model(torch.from_numpy(X))
```

## Reusing the loader in modeling code

```python
from src.eda import load_mnist

X_train, y_train, X_test, y_test = load_mnist("data/archive (1)")
```
