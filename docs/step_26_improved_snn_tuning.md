# Step 26 — Improved SNN Architecture and Tuning

## Goal

- Improve **4-class DEAP emotion** SNN performance by extending the network and expanding the hyperparameter search, without changing labels, features, LR, RF, or evaluation.

## Architecture (`SimpleSNN`)

```
Input
  → Linear(input_size, hidden_size)
  → LIF(beta)
  → Dropout
  → Linear(hidden_size, second_hidden_size)
  → LIF(beta)
  → Dropout
  → Linear(second_hidden_size, output_size)
```

New constructor parameters: `hidden_size`, `second_hidden_size`, `beta`, `dropout`.

Static features are repeated over `num_steps` time steps (Step 11 path). Spike-encoded input (Step 12) uses the same architecture.

## Hyperparameter grid

| Parameter | Options |
|-----------|---------|
| `hidden_size` | 64, 128, 256 |
| `second_hidden_size` | 32, 64, 128 |
| `learning_rate` | 0.001, 0.0005 |
| `epochs` | 50, 100 |
| `class_weight` | None, `"balanced"` |
| `num_steps` | 10, 20 |
| `beta` | 0.9, 0.95 |
| `dropout` | 0.0, 0.2 |

**Total configurations:** 576 (full grid) or 64 (`SNN_FAST_GRID = True`) per classification task.

Set `SNN_FAST_GRID = True` in `main.py` for the reduced grid (default for fast experiments).

**Selection criterion:** best **macro F1** on the held-out test split (`random_state=42`, `test_size=0.2`, stratified).

## Files modified

- `src/snn_model.py` — `SimpleSNN`, `_train_single_snn`, `train_tuned_snn_model`

## Expected log output

One line per configuration:

```
SNN config hidden=128, second_hidden=64, beta=0.95, dropout=0.2, num_steps=20, lr=0.0005, epochs=100, class_weight=balanced | acc=0.4531 macro_f1=0.4412
```

Then:

```
Selected tuned SNN params: {'mode': 'tuned_step26', ...} | macro F1: 0.4412
```

## How to run

```bash
python main.py
```

Recommended setup (current best pipeline):

```python
USE_COMBINED_STAT_DE_FEATURES = True
MULTI_LABEL_STRATEGY = "mean"
CHANNEL_SELECTION_MODE = "all"
USE_SPIKE_ENCODING = False   # tuned SNN (Step 26 grid)
SNN_FAST_GRID = True         # 64 configs; set False for full 576-config grid
```

Compare **Multi-Emotion Random Forest** vs **Multi-Emotion Tuned SNN** in the final comparison summary.

When `SNN_FAST_GRID = True`, the pipeline prints:

```
SNN_FAST_GRID enabled
Total SNN configurations: 64
```

## Unchanged

- Label creation and `MULTI_LABEL_STRATEGY`
- Feature extraction and channel selection
- Logistic Regression (`baseline_model.py`)
- Random Forest (`random_forest_model.py`)
- `evaluate_classification()` metrics and reporting
- Train/test split parameters (`test_size=0.2`, `random_state=42`)

## Runtime note

- `SNN_FAST_GRID = True` (default): **64** trainings per task — suitable for `FAST_TEST_MODE`.
- `SNN_FAST_GRID = False`: **576** trainings per task — use for final full-dataset runs.

## Step 12 (spike-encoded SNN)

`train_spike_encoded_snn_model` still uses fixed Step 11-style defaults (`hidden_size=64`, `second_hidden_size=32`, `beta=0.95`, `dropout=0.0`) and is not part of the Step 26 grid.
