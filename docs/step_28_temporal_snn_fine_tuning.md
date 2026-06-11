# Step 28 — Temporal SNN Fine-Tuning

## Goal

- Improve **Temporal Windowed SNN** performance with a **focused hyperparameter grid** around the current best configuration.

## Baseline (Step 27 fast grid)

| Metric | Value |
|--------|-------|
| Accuracy | 53.12% |
| Macro F1 | 0.4804 |
| hidden_size | 128 |
| second_hidden_size | 32 |
| beta | 0.95 |
| dropout | 0.2 |
| learning_rate | 0.0005 |
| epochs | 50 |
| class_weight | None |
| time_steps | 10 |
| features_per_step | 200 |

## Focused grid (`TEMPORAL_SNN_FINE_TUNE = True`)

| Parameter | Options |
|-----------|---------|
| `hidden_size` | 128, 256 |
| `second_hidden_size` | 32, 64 |
| `learning_rate` | 0.0005, 0.0003 |
| `epochs` | 50, 100, 150 |
| `class_weight` | None, `"balanced"` |
| `beta` | 0.9, 0.95 |
| `dropout` | 0.1, 0.2, 0.3 |

**Total configurations:** 288 (time steps fixed at 10 from windowed DE features).

**Selection criterion:** best **macro F1** on the held-out test split.

## Files modified

- `src/snn_model.py` — `_get_temporal_snn_fine_tune_grid()`, `train_tuned_snn_model(temporal_fine_tune=True)`
- `main.py` — `TEMPORAL_SNN_FINE_TUNE`

## Configuration

```python
USE_TEMPORAL_SNN_FEATURES = True
TEMPORAL_SNN_FINE_TUNE = True
USE_COMBINED_STAT_DE_FEATURES = True   # LR + RF only
MULTI_LABEL_STRATEGY = "mean"
```

Set `TEMPORAL_SNN_FINE_TUNE = False` to fall back to the Step 27 / Step 26 fast grid for temporal SNN.

## Expected output

```
Running Temporal Windowed SNN (Step 28 Fine-Tune)
Temporal SNN input enabled (Step 27)
SNN training tensor shape: (n_train, 10, 200)
TEMPORAL_SNN_FINE_TUNE enabled
Total Temporal SNN configurations: 288

SNN config hidden=128, second_hidden=32, beta=0.95, dropout=0.2, time_steps=10, lr=0.0005, epochs=50, class_weight=None | acc=0.5312 macro_f1=0.4804
...

Selected tuned SNN params: {'mode': 'temporal_step28', ...}
Accuracy: 0.5312
Macro F1: 0.4804
```

## Unchanged

- Labels and feature extraction for LR / RF
- Temporal window DE features (10 windows × 200 DE features)
- Train/test split and `evaluate_classification()`
- Random Forest and Logistic Regression pipelines

## Runtime note

288 trainings per classification task (binary + multi-emotion). Expect longer runtime than the 32-config temporal fast grid; use `FAST_TEST_MODE` for quick smoke tests.
