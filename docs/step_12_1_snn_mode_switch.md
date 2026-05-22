# Step 12.1 — Configurable SNN Mode Selection

## Goal

- Keep **both** SNN implementations (Step 11 tuned model and Step 12 spike-encoded model) in the project and switch between them with a single configuration flag in `main.py`.

## Available modes

| Mode | Flag | Function | Description |
|------|------|----------|-------------|
| **Tuned SNN (default)** | `USE_SPIKE_ENCODING = False` | `train_tuned_snn_model` | Step 11 grid search over hyperparameters; best config by macro F1; static scaled features |
| **Spike-encoded SNN (experimental)** | `USE_SPIKE_ENCODING = True` | `train_spike_encoded_snn_model` | Step 12 rate-encoded spike trains; fixed best hyperparameters from Step 11 |

Both use the same `SimpleSNN` class; forward supports static input (Step 11) or spike tensor input (Step 12).

## How to switch modes

Edit `main.py` near the top:

```python
USE_SPIKE_ENCODING = False   # Step 11 (default)
# USE_SPIKE_ENCODING = True  # Step 12 (experimental)
```

Then run:

```bash
python main.py
```

## Default configuration

```python
USE_SPIKE_ENCODING = False
```

Step 11 tuned SNN is the **default** main model. Step 12 remains available for experiments.

## Expected behavior

### `USE_SPIKE_ENCODING = False`

- Prints: `Running Tuned SNN (Step 11)`
- Runs hyperparameter grid search (16 configurations)
- Prints selected params and macro F1
- Evaluation label: **Tuned SNN**
- Comparison summary includes `Active SNN mode: tuned_step11`

### `USE_SPIKE_ENCODING = True`

- Prints: `Running Spike-Encoded SNN (Step 12)`
- Trains with rate encoding (`num_steps=10`)
- Loss every 10 epochs
- Evaluation label: **Spike-encoded SNN**
- Comparison summary includes `Active SNN mode: spike_encoded_step12`

### Unchanged in both modes

- DEAP loading, preprocessing, feature extraction, binary arousal labels
- Baseline Logistic Regression tuning and evaluation
- `evaluate_classification` and final comparison summary (baseline + active SNN)

## Notes

- No SNN code was removed; Step 11 and Step 12 live in `src/snn_model.py` as separate training functions.
- `train_snn_model` is an alias for `train_spike_encoded_snn_model` (backward compatibility).
- Spike-encoded mode can take longer per run depending on data size; tuned mode runs a full grid search and may take even longer on many subjects.
