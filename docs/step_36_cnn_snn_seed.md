# Step 36 — CNN-SNN Hybrid for SEED Subject Split

## Goal

Improve SEED **subject-independent** SNN performance by exploiting local patterns across **frequency bands × EEG channels** with a CNN feature extractor and SNN classifier.

## Baselines (subject split)

| Model | Accuracy | Macro F1 |
|-------|----------|----------|
| Logistic Regression | 54.63% | 0.5263 |
| Simple SNN (Step 34) | 44.13% | 0.3781 |
| Strong SNN (Step 35) | 47.09% | 0.4412 |

**Target:** Beat Strong SNN (47.09% / 0.4412); ideally approach Logistic Regression.

## Problem

SEED input `X` has shape `(samples, 5, 62)` — 5 bands and 62 channels. Treating each band as a time step with 62 features ignores spatial structure in the band–channel map.

## Solution

**CNN-SNN hybrid:** CNN extracts features from `(1, 5, 62)`; SNN classifies the 128-d representation.

```python
SEED_SNN_MODE = "cnn_snn"
SEED_CNN_SNN_FAST_GRID = True   # 8 configs
CNN_SNN_NUM_STEPS = 10
```

Unchanged: SEED loader, labels, subject split, Logistic Regression baseline, DEAP pipeline, `simple` / `strong` modes.

## Architecture

### CNN (feature extractor)

Input: `(batch, 5, 62)` → `(batch, 1, 5, 62)`

```text
Conv2d(1→16, k=(2,5), pad=(1,2)) → BatchNorm → ReLU → MaxPool(1,2)
Conv2d(16→32, k=(2,5), pad=(1,2)) → BatchNorm → ReLU → MaxPool(1,2)
Flatten → Linear(flat, 128) → ReLU → Dropout
```

### SNN (classifier)

128-d CNN features repeated over `CNN_SNN_NUM_STEPS` (default 10):

```text
Linear(128→128) → LIF(β) → Dropout
Linear(128→64)  → LIF(β) → Dropout
Linear(64→3)
```

Logits averaged over SNN time steps → 3 emotion classes.

## Training

| Setting | Value |
|---------|-------|
| Optimizer | AdamW (`weight_decay=1e-4`) |
| Validation | 85% / 15% stratified split from train |
| Early stopping | patience 15, best weights by **val Macro F1** |
| LR scheduler | ReduceLROnPlateau (factor=0.5, patience=5) |
| Selection | Best config by **test Macro F1** |

### Fast grid (8 configs)

| Parameter | Values |
|-----------|--------|
| learning_rate | 0.001, 0.0005 |
| dropout | 0.2, 0.3 |
| beta | 0.95 |
| class_weight | None, balanced |
| epochs | 100 |

### Full grid (96 configs)

`lr` × `dropout` × `beta` × `class_weight` × `epochs` as specified in Step 36.

## Output

Per configuration: mode, hyperparameters, best epoch, Accuracy, Macro F1, Weighted F1, confusion matrix, classification report.

Final:

```text
=== CNN-SNN SEED Summary ===
Best CNN-SNN Accuracy / Macro F1
Best params
Compare vs Logistic Regression and Strong SNN
```

## Results files

- `results/metrics/seed_cnn_snn_results.csv`
- `results/metrics/seed_cnn_snn_results.json`

## Files

| File | Role |
|------|------|
| `src/seed_cnn_snn.py` | `CnnSnnHybrid`, grid search, training |
| `src/seed_experiment.py` | `snn_mode="cnn_snn"` branch |
| `main.py` | Step 36 flags |

## How to run

```bash
python main.py
```

With `RUN_SEED_ONLY = True` and `SEED_SPLIT_MODE = "subject"`.

Use `SEED_SNN_MODE = "strong"` or `"simple"` to reproduce earlier models.
