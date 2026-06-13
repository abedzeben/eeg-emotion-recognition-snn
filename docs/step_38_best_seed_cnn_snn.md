# Step 38 — Best SEED CNN-SNN Normalization (Full Run)

## Goal

Run the **best normalization from Step 37** in full mode with complete evaluation.

## Best normalization

**`per_subject_per_channel`**

Step 37 fast result (30 epochs):

| Metric | Value |
|--------|-------|
| Accuracy | 69.07% |
| Macro F1 | 0.6937 |

## Configuration

```python
RUN_SEED_EXPERIMENT = True
RUN_SEED_ONLY = True
SEED_RUN_BEST_NORMALIZATION_ONLY = True
SEED_SNN_MODE = "cnn_snn"
SEED_SPLIT_MODE = "subject"
CNN_SNN_NUM_STEPS = 10
```

### Fixed settings

| Setting | Value |
|---------|-------|
| Normalization | `per_subject_per_channel` only |
| Split | Subject (train 0–11, test 12–14) |
| Epochs | 100 |
| CNN-SNN config | Step 36/37 best (lr=0.001, dropout=0.3, beta=0.95, balanced) |

No other normalization modes. No DEAP. Architecture unchanged.

## Output

Full metrics per run:

- Accuracy, Macro F1, Weighted F1
- Confusion matrix
- Classification report
- Per-class recall

Summary vs Step 37 fast best, Logistic Regression, and previous global CNN-SNN.

## Results files

- `results/metrics/seed_best_cnn_snn_results.csv`
- `results/metrics/seed_best_cnn_snn_results.json`

## Files

| File | Role |
|------|------|
| `src/seed_subject_shift_study.py` | `run_seed_best_cnn_snn()` |
| `main.py` | `SEED_RUN_BEST_NORMALIZATION_ONLY` |

## How to run

```bash
python main.py
```

To run Step 37 study instead:

```python
SEED_RUN_BEST_NORMALIZATION_ONLY = False
RUN_SEED_SUBJECT_SHIFT_STUDY = True
```
