# MNIST Digit Classification

Handwritten-digit classification on the classic MNIST dataset, packaged as a
production-style, modular project: a Keras/TensorFlow fully-connected ANN (MLP)
with a complete **training + evaluation pipeline**, a **FastAPI** inference
backend, and a **clean draw-a-digit web UI**.

The trained MLP reaches **~98% test accuracy** (the current run is 98.09%).

## Architecture

```
mnist_classifier/                # importable library — the source of truth
├── config.py                    # one Config dataclass (env-overridable) for everything
├── data/
│   ├── idx_loader.py            # robust IDX/ubyte loader (flat + nested layouts)
│   └── preprocess.py            # flatten + normalize + stratified split
├── models/
│   └── mlp.py                   # Keras Sequential MLP factory
├── training/
│   ├── train.py                 # end-to-end training pipeline  -> train(config)
│   ├── evaluate.py              # retrain-free evaluation        -> evaluate(config)
│   └── reporting.py             # shared figures + markdown reports
└── inference/
    └── predictor.py             # load-once Predictor for serving (handles any image)

scripts/                         # thin CLI entrypoints over the library
├── train.py                     # python -m scripts.train
└── evaluate.py                  # python -m scripts.evaluate

api/                             # FastAPI backend
├── main.py                      # /health, /predict, /predict/canvas, / (UI)
└── schemas.py                   # pydantic request/response models

ui/index.html                    # canvas draw-a-digit UI (served by the API)

src/eda/                         # exploratory data analysis pipeline (unchanged)
data/archive (1)/                # raw MNIST IDX files (tracked in git)
model_outputs/                   # trained model + metrics + figures (gitignored)
eda_outputs/                     # EDA report + figures (gitignored)
```

Why this shape: each layer has one job and no circular knowledge. The same
`config.Config` drives the library, CLI, and API; the same `Predictor`
preprocessing serves every caller, so there is no train/serve skew. Training is
separate from evaluation so model quality can be re-measured or compared across
runs without retraining, and inference is separate from both so the API never
imports the training loop.

## Setup

```bash
pip install -r requirements.txt
```

Requires Python 3.9+ with `tensorflow` (Keras 3), `scikit-learn`, `matplotlib`,
`pillow`, `fastapi`, and `uvicorn`.

## Data

The four standard MNIST IDX files live under `data/archive (1)/` (the loader
tolerates both the flat and the nested Kaggle-archive folder layout) and are
committed to the repo, so everything runs out of the box after cloning:

- `train-images.idx3-ubyte`, `train-labels.idx1-ubyte`
- `t10k-images.idx3-ubyte`, `t10k-labels.idx1-ubyte`

## 1. Explore the data (optional but recommended)

```bash
python -m src.eda.run_eda --tsne
```

Writes `eda_outputs/EDA_REPORT.md` + figures (class balance, ink profiles,
duplicates/leakage, PCA/t-SNE, outliers, modeling recommendations).

## 2. Train

```bash
# Defaults: MLP 784->256->128->10, ReLU + dropout 0.2 + L2 1e-4,
# Adam lr 1e-3, batch 128, up to 30 epochs, early stopping on val loss.
python -m scripts.train

# Override any hyperparameter (every flag maps to a Config field)
python -m scripts.train --hidden 512,256,128 --dropout 0.3 --epochs 50
python -m scripts.train --data-dir "data/archive (1)" --out-dir model_outputs
```

The pipeline: flatten 28×28→784 + normalize → stratified 10% val split → build &
compile the MLP → fit with early stopping (best weights restored) → evaluate on
val and the held-out test set. Writes to `model_outputs/`:
`best_model.keras`, `metrics.json`, `training_history.csv`,
`TRAINING_REPORT.md`, and `figures/{learning_curves,confusion_matrix}.png`.

Every knob can also be set via `MNIST_`-prefixed env vars (e.g. `MNIST_EPOCHS=50`).

## 3. Evaluate

```bash
python -m scripts.evaluate
```

Loads `best_model.keras` (no retraining) and writes `EVALUATION_REPORT.md`,
`evaluation.json`, and `figures/{confusion_matrix,misclassified}.png` —
per-class precision/recall/F1, macro/weighted averages, ranked confusions, and a
gallery of the actual misclassified digits.

## 4. Serve the API + UI

```bash
uvicorn api.main:app --reload          # http://127.0.0.1:8000
# or: python -m api.main
```

The model loads once at startup. Open `http://127.0.0.1:8000/` for the
draw-a-digit UI, or `http://127.0.0.1:8000/docs` for interactive OpenAPI docs.

| Method | Path              | Body                                   | Returns |
|--------|-------------------|----------------------------------------|---------|
| GET    | `/health`         | —                                      | liveness + whether the model is loaded |
| POST   | `/predict`        | multipart image file (any size)        | `{digit, confidence, probabilities}` |
| POST   | `/predict/canvas` | JSON `{pixels: [...], auto_orient}`     | `{digit, confidence, probabilities}` |
| GET    | `/`               | —                                      | the web UI |

`/predict` accepts a PNG/JPG of any size (grayscaled + resized server-side).
`/predict/canvas` takes a row-major grayscale array (784 values, or any N²
square that the server downsamples). `Predictor` auto-orients dark-on-light
input to MNIST's white-on-black convention.

Example:

```bash
curl -F "file=@my_digit.png" http://127.0.0.1:8000/predict
```

## Using the library directly

```python
from mnist_classifier import Predictor, load_config

predictor = Predictor.from_config(load_config())
result = predictor.predict(image_28x28_or_any_size)   # numpy array
print(result.digit, result.confidence)                # e.g. 7, 0.998
```

```python
# Or run the pipelines programmatically
from mnist_classifier.config import load_config
from mnist_classifier.training import train, evaluate

cfg = load_config(epochs=50, dropout=0.3)
train(cfg)
evaluate(cfg)
```

## Notes

- `model_outputs/` is gitignored — the model is a build artifact. **Train before
  serving**, or the API stays up but `/predict*` returns 503 (and `/health`
  reports `model_loaded: false`).
- An MLP plateaus around ~98% on MNIST; a CNN is the next step past ~99%. Design
  rationale for the hyperparameters lives in
  `.claude/skills/mnist-ann/references/ann_design_notes.md`.
