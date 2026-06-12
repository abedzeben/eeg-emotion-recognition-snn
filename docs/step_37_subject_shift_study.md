# Step 37 — Subject Shift Normalization Study for CNN-SNN

## Goal

Test whether **subject-specific normalization** can close the gap between CNN-SNN and Logistic Regression on SEED **subject-independent** evaluation.

## Background

| Model | Accuracy | Macro F1 |
|-------|----------|----------|
| Logistic Regression | 54.63% | 0.5263 |
| Strong SNN | 47.09% | 0.4412 |
| CNN-SNN (Step 36) | 48.79% | 0.4912 |

**Hypothesis:** Residual error is partly due to **subject shift** — EEG feature distributions differ across subjects and global normalization may not remove this sufficiently.

## What changes

**Only normalization** is varied. Fixed:

- SEED loader, labels, subject split (train 0–11, test 12–14)
- CNN-SNN architecture and hyperparameters
- Logistic Regression baseline (reference numbers only)
- DEAP pipeline

```python
RUN_SEED_SUBJECT_SHIFT_STUDY = True
SEED_SNN_MODE = "cnn_snn"
SEED_SPLIT_MODE = "subject"
RUN_SEED_ONLY = True
```

### Fixed CNN-SNN config (no grid search)

| Parameter | Value |
|-----------|-------|
| learning_rate | 0.001 |
| dropout | 0.3 |
| beta | 0.95 |
| class_weight | balanced |
| epochs | 100 |

## Normalization modes

| Mode | Description |
|------|-------------|
| `none` | Raw features |
| `global` | StandardScaler fit on **training data only**, transform train & test |
| `per_subject` | Per subject: one mean/std over all bands and channels |
| `per_subject_per_channel` | Per subject and channel: mean/std over bands and trials |
| `per_band` | StandardScaler per frequency band (fit on train) |
| `per_subject_per_band` | Per subject and band: mean/std over channels and trials |

Subject-aware test normalization uses **each test subject’s own statistics** (realistic when subject ID is known at inference).

## Output

Per mode: Accuracy, Macro F1, Weighted F1, confusion matrix, classification report.

Summary:

```text
=== Subject Shift Study Summary ===
normalization | accuracy | macro_f1
(sorted by macro_f1)

Best normalization / Accuracy / Macro F1
Compare vs Logistic Regression and previous CNN-SNN
Improvement deltas
```

## Results files

- `results/metrics/seed_subject_shift_study.csv`
- `results/metrics/seed_subject_shift_study.json`

## Files

| File | Role |
|------|------|
| `src/seed_subject_shift_study.py` | Normalization + study runner |
| `src/seed_cnn_snn.py` | `BEST_CNN_SNN_CONFIG`, `train_cnn_snn_fixed_config()` |
| `main.py` | `RUN_SEED_SUBJECT_SHIFT_STUDY` |

## How to run

```bash
python main.py
```

Set `RUN_SEED_SUBJECT_SHIFT_STUDY = False` to run the standard Step 36 CNN-SNN grid instead.

## Fast mode

```python
SEED_SUBJECT_SHIFT_FAST = True
```

When enabled:

- Tests only: `global`, `per_subject_per_channel`, `per_subject_per_band`
- Uses `epochs = 30` (other CNN-SNN hyperparameters unchanged)
- Prints compact metrics per mode (no full classification report)
- Same summary table and export files at the end

Set `SEED_SUBJECT_SHIFT_FAST = False` for the full 6-mode study with 100 epochs.

## Success criterion

Determine whether any normalization mode **significantly improves** CNN-SNN Macro F1 over 0.4912 and approaches Logistic Regression (0.5263) **without** changing architecture or hyperparameters.
