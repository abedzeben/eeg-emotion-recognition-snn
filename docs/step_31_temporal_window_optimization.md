# Step 31 — Temporal Window Count Optimization

## Goal

- Find the best **number of EEG temporal windows** for the Temporal SNN by comparing multiple window counts with the fixed best hyperparameter configuration.

## Baseline

| Setting | Value |
|---------|-------|
| Windows | 10 |
| Accuracy | 53.12% |
| Macro F1 | 0.5103 |

## Method

For each value in `TEMPORAL_WINDOW_OPTIONS`:

1. Extract windowed DE features: `(trials, num_windows, 200)`
2. Train **one** Temporal SNN with the Step 28/29 best config (no grid search)
3. Record accuracy and macro F1
4. Select the window count with **highest macro F1**

| Window count | Features per step | Example shape (320 trials) |
|--------------|-------------------|----------------------------|
| 5 | 200 | `(320, 5, 200)` |
| 10 | 200 | `(320, 10, 200)` |
| 20 | 200 | `(320, 20, 200)` |
| 40 | 200 | `(320, 40, 200)` |

`window_size = trial_length // num_windows` (DEAP: 8064 samples per trial).

## Configuration

```python
RUN_TEMPORAL_WINDOW_OPTIMIZATION = True
TEMPORAL_WINDOW_OPTIONS = [5, 10, 20, 40]

USE_TEMPORAL_SNN_FEATURES = True
USE_BEST_TEMPORAL_SNN_CONFIG = True
TEMPORAL_SPIKE_ENCODING = False   # required for fair window comparison
RUN_CLASSICAL_MODELS = False
MULTI_LABEL_STRATEGY = "mean"
```

Set `RUN_TEMPORAL_WINDOW_OPTIMIZATION = False` to restore single-window training with `TEMPORAL_NUM_WINDOWS = 10`.

## Best SNN config (fixed per run)

| Parameter | Value |
|-----------|-------|
| hidden_size | 128 |
| second_hidden_size | 32 |
| beta | 0.95 |
| dropout | 0.2 |
| learning_rate | 0.0005 |
| epochs | 50 |
| class_weight | None |

## Expected output

```
=== Temporal window optimization (Step 31) ===
Temporal window options: [5, 10, 20, 40]

--- Temporal windows: 10 ---
Number of windows: 10
Feature shape: (320, 10, 200)
Accuracy: 0.5312
Macro F1: 0.5103

=== Best temporal window count (Step 31) ===
Selected by Macro F1
Number of windows: ...
```

## Files modified

- `main.py` — `_run_temporal_window_optimization()`, flags

## Unchanged

- `MULTI_LABEL_STRATEGY = "mean"` labeling
- Train/test split inside `train_tuned_snn_model`
- SNN architecture and evaluation
- No Logistic Regression, Random Forest, or grid search during optimization

## Runtime note

Each window count trains one full SNN (50 epochs). With 4 options, expect ~4× the Step 29 runtime.
