# Step 27 — Temporal Windowed SNN Input

## Goal

- Improve SNN performance by feeding **real temporal EEG dynamics** instead of repeating a static feature vector over `num_steps`.

## Problem (Step 11–26)

The tuned SNN received a single combined feature vector and **repeated it** across time steps. That does not exploit sequential EEG structure or the temporal processing strength of SNNs.

## Solution

For **SNN only**, extract **windowed Differential Entropy** features:

1. Split each trial into **10 fixed windows** (`window_size = trial_length // 10`).
2. Per window, compute **DE** for delta, theta, alpha, beta, gamma on **all 40 channels**.
3. Feed `(batch, time_steps, features)` into the SNN — one distinct vector per window, processed sequentially.

| Property | Value |
|----------|-------|
| Time steps | 10 |
| Features per step | 200 (40 × 5 DE bands) |
| Output shape | `(num_trials, 10, 200)` |

Classical models (Logistic Regression, Random Forest) still use **combined Statistical + DE** flat features (440 dims).

## Files modified

- `src/features.py` — `extract_temporal_window_de_features()`, `print_temporal_snn_feature_info()`
- `src/snn_model.py` — temporal `SimpleSNN.forward()`, `_scale_temporal_features()`, `train_tuned_snn_model(temporal=True)`
- `main.py` — `USE_TEMPORAL_SNN_FEATURES`

## Configuration

```python
USE_TEMPORAL_SNN_FEATURES = True
USE_COMBINED_STAT_DE_FEATURES = True   # LR + RF only
CHANNEL_SELECTION_MODE = "all"
MULTI_LABEL_STRATEGY = "mean"
SNN_FAST_GRID = True
```

Temporal features are extracted from **all 40 channels** before channel selection is applied to classical-model inputs.

## Expected output

```
=== Temporal SNN features (Step 27) ===
Temporal SNN feature shape: (num_trials, 10, 200)
SNN input per time step: 200
Number of time steps: 10

Running Temporal Windowed SNN (Step 27)
Temporal SNN input enabled (Step 27)
SNN training tensor shape: (n_train, 10, 200)
```

## SNN forward pass (temporal mode)

```
For t in 0..9:
  x_t = input[:, t, :]          # 200 DE features for window t
  → Linear → LIF → Dropout → Linear → LIF → Dropout → Linear
  accumulate output spikes
Return mean output over 10 windows
```

No static-vector repetition in this mode. `num_steps` grid search is skipped; time steps are fixed by the number of windows.

## How to compare

```bash
python main.py
```

| SNN mode | Flag | Input |
|----------|------|-------|
| Static (Step 26) | `USE_TEMPORAL_SNN_FEATURES = False` | `(trials, 440)` repeated |
| Temporal (Step 27) | `USE_TEMPORAL_SNN_FEATURES = True` | `(trials, 10, 200)` sequential |

Compare **Multi-Emotion SNN Macro F1** against the previous best static SNN result.

## Unchanged

- Labels and `MULTI_LABEL_STRATEGY`
- Combined Stat+DE features for LR and RF
- Train/test split (`test_size=0.2`, `random_state=42`)
- `evaluate_classification()` reporting
- Step 26 hyperparameter grid (minus `num_steps` when temporal)
