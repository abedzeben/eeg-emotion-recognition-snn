# Step 40 — DEAP Temporal SNN Normalization Study

## Goal

Test whether **subject-aware normalization** improves the original DEAP **Temporal SNN** without CNN or feature reshaping.

## Background

Historical DEAP Temporal SNN baseline:

| Metric | Value |
|--------|-------|
| Accuracy | ~53.12% |
| Macro F1 | ~0.5103 |

SEED improved with `per_subject_per_channel` normalization. This step tests the same idea on DEAP using the **existing** temporal representation.

## Configuration

```python
RUN_DEAP_TEMPORAL_NORMALIZATION_STUDY = True
MULTI_LABEL_STRATEGY = "mean"
USE_TEMPORAL_SNN_FEATURES = True
USE_BEST_TEMPORAL_SNN_CONFIG = True
TEMPORAL_SPIKE_ENCODING = False
RUN_CLASSICAL_MODELS = False
```

## What runs

**Temporal SNN only** — best fixed config (Step 29):

- hidden=128, second_hidden=32, beta=0.95, dropout=0.2, lr=0.0005, epochs=50

**Feature shape:** `(trials, 10, 200)` — 40 channels × 5 DE bands per window.

No CNN. No 5×40 reshape for the model. No SEED, LR, RF, or binary task.

## Normalization modes tested

| Mode | Description |
|------|-------------|
| `global` | StandardScaler on training data (per feature) |
| `per_subject` | Z-score per subject (all windows and features) |
| `per_subject_per_channel` | Z-score per subject and EEG channel (5 bands grouped per channel) |

Train statistics are used for test subjects when available (same split as Step 39).

No extra StandardScaler is applied after normalization during training.

## Output

Per mode:

- Normalization mode
- Accuracy
- Macro F1
- Confusion matrix

Summary table sorted by Macro F1 vs historical baseline.

## Run

```python
RUN_DEAP_TEMPORAL_NORMALIZATION_STUDY = True
RUN_DEAP_CNN_SNN = False
RUN_SEED_EXPERIMENT = False
```

```bash
python main.py
```

## Files

| File | Role |
|------|------|
| `src/deap_temporal_normalization_study.py` | Study runner |
| `src/deap_cnn_snn.py` | `train_deap_temporal_baseline(apply_standard_scaler=False)` |
| `main.py` | Step 40 flag |

## Success criterion

Determine whether `per_subject_per_channel` (or another mode) improves Temporal SNN Macro F1 beyond ~0.5103 without CNN.
