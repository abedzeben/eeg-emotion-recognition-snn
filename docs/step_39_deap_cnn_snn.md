# Step 39 — DEAP CNN-SNN with Subject-Aware Normalization

## Goal

Apply the successful SEED strategy to DEAP and test whether CNN-SNN with **per_subject_per_channel** normalization improves over the current Temporal SNN baseline (~53% / 0.51 Macro F1).

## SEED strategy (reference)

1. `per_subject_per_channel` normalization
2. CNN on `(bands × channels)` map before SNN
3. Subject-aware evaluation on SEED

## DEAP representation

| Stage | Shape |
|-------|-------|
| Original temporal DE | `(trials, 10, 200)` |
| Reshaped | `(trials, 10, 5, 40)` |

200 = 40 channels × 5 bands → `(5 bands, 40 channels)` per window.

## Normalization modes

`DEAP_NORMALIZATION_MODE`:

| Mode | Description |
|------|-------------|
| `none` | Raw features |
| `global` | StandardScaler on training data |
| `per_subject` | Z-score per subject |
| `per_subject_per_channel` | Z-score per subject and channel (train stats; default) |

## CNN-SNN architecture

Per window `(5, 40)` → `(1, 5, 40)`:

```text
Conv2D → BatchNorm → ReLU
Conv2D → BatchNorm → ReLU
AdaptiveAvgPool → Linear → 128 features
→ Temporal SNN over 10 windows → 4 classes
```

## Comparison

Runs on the **same train/test split**:

1. **Temporal SNN** — best fixed config (Step 29)
2. **CNN-SNN** — SEED-style config (lr=0.001, dropout=0.3, balanced, 100 epochs)

Prints `=== DEAP Comparison ===` with accuracy/Macro F1 deltas.

## Run control flags (`main.py`)

Three independent modes:

```python
RUN_DEAP_CNN_SNN = True
RUN_SEED_BEST_MODEL = False
RUN_FINAL_DATASET_COMPARISON = False
DEAP_NORMALIZATION_MODE = "per_subject_per_channel"
```

| Flag | Behavior |
|------|----------|
| `RUN_DEAP_CNN_SNN = True` | DEAP CNN-SNN experiment only; no SEED |
| `RUN_SEED_BEST_MODEL = True` | Step 38 best SEED model only; no DEAP |
| `RUN_FINAL_DATASET_COMPARISON = True` | Load saved JSON results; no retraining |

Only one primary mode should be `True` at a time.

### Examples

**DEAP only:**

```python
RUN_DEAP_CNN_SNN = True
RUN_SEED_BEST_MODEL = False
RUN_FINAL_DATASET_COMPARISON = False
```

**SEED best only:**

```python
RUN_SEED_BEST_MODEL = True
```

**Compare datasets (after both runs):**

```python
RUN_FINAL_DATASET_COMPARISON = True
```

Requires:

- `results/metrics/seed_best_cnn_snn_results.json`
- `results/metrics/deap_cnn_snn_results.json`

## Results files

- `results/metrics/deap_cnn_snn_results.csv`
- `results/metrics/deap_cnn_snn_results.json`

## Files

| File | Role |
|------|------|
| `src/deap_cnn_snn.py` | Reshape, normalize, models, experiment, final comparison |
| `main.py` | Step 39 flags and run routing |

## Success criterion

Determine whether **per_subject_per_channel + 5×40 CNN + SNN** exceeds DEAP baseline **~53.12% accuracy / 0.5103 Macro F1**.

SEED code is unchanged. SEED is not run when `RUN_DEAP_CNN_SNN = True`.
