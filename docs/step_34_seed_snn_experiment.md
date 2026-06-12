# Step 34 — SEED Dataset SNN Experiment

## Goal

Test whether the **Temporal SNN** achieves higher emotion-recognition accuracy on the **SEED** dataset than on **DEAP**.

SEED provides cleaner 3-class emotion labels (negative / neutral / positive), while DEAP uses a noisy Valence–Arousal mapping to 4 classes (~53% accuracy / 0.51 Macro F1).

## Dataset

Place three NPZ files in `data/seed/`:

| File | Key | Shape |
|------|-----|-------|
| `DatasetCaricatoNoImage.npz` | `arr_0` | `(50910, 5, 62)` |
| `LabelsNoImage.npz` | `arr_0` | `(50910,)` |
| `SubjectsNoImage.npz` | `arr_0` | `(50910,)` |

| Dimension | Meaning |
|-----------|---------|
| 50910 | Samples |
| 5 | Frequency bands (time steps for SNN) |
| 62 | EEG channels (features per step) |
| 3 | Emotion classes (0, 1, 2) |
| 15 | Subjects (0–14) |

**No DEAP label logic** is used for SEED.

## Configuration (`main.py`)

```python
RUN_SEED_EXPERIMENT = True
SEED_DATA_DIR = "data/seed"
SEED_SPLIT_MODE = "trial"              # "trial" | "subject"
SEED_NORMALIZATION_MODE = "train_only_standard"  # "global" | "train_only_standard"
```

The DEAP pipeline is unchanged and runs after SEED when `data/raw` contains DEAP files.

## Temporal SNN input

SEED features are used directly as temporal input:

```text
X shape: (samples, 5, 62)
  time_steps = 5 frequency bands
  features_per_step = 62 EEG channels
```

Same `SimpleSNN` temporal mode as DEAP (`forward(..., temporal=True)`).

### SEED SNN config

| Parameter | Value |
|-----------|-------|
| hidden_size | 128 |
| second_hidden_size | 64 |
| beta | 0.95 |
| dropout | 0.2 |
| learning_rate | 0.0005 |
| epochs | 50 |
| batch_size | 128 |
| num_classes | 3 |

## Split modes

### `trial`

- `train_test_split`, `test_size=0.2`, `random_state=42`, `stratify=y`
- Random trial split (may mix subjects in train/test)

### `subject`

- Train: subjects **0–11**
- Test: subjects **12–14**
- Subject-independent evaluation (more realistic generalization)

## Normalization

| Mode | Description |
|------|-------------|
| `global` | Fit `StandardScaler` on full dataset, then split |
| `train_only_standard` | Split first; fit scaler on **train** only; transform train and test |

3D data is reshaped to 2D for scaling and reshaped back to `(samples, 5, 62)`.

## Baseline

Simple **Logistic Regression** on flattened features `X.reshape(samples, 5*62)` — fixed `C=1.0`, no grid search. For comparison only; **SNN remains the main model**.

## Evaluation output

For baseline and SNN:

- Accuracy
- Macro F1
- Weighted F1
- Confusion matrix
- Classification report

Final summary:

```text
=== SEED Final Comparison ===
Baseline Accuracy / Macro F1
SNN Accuracy / Macro F1
```

## Results files

- `results/metrics/seed_results_summary.csv`
- `results/metrics/seed_results_summary.json`

## Files added / modified

| File | Role |
|------|------|
| `src/seed_experiment.py` | Load, split, normalize, train, evaluate, export |
| `src/snn_model.py` | `SEED_SNN_CONFIG`, `train_seed_snn_model()` |
| `main.py` | Step 34 flags and `run_seed_experiment()` call |

## How to run

```bash
# Place NPZ files in data/seed/
python main.py
```

SEED-only (skip DEAP if no `.dat` files):

```python
RUN_SEED_EXPERIMENT = True
RUN_SNN_RESEARCH_EXPERIMENTS = False  # optional: skip long DEAP grid
```

Subject-independent evaluation:

```python
SEED_SPLIT_MODE = "subject"
```

## Expected outcome

If SEED SNN accuracy and Macro F1 are **significantly higher** than DEAP (~53% / 0.51), the cleaner labels and pre-extracted band features likely help the SNN. Subject split results are the stronger generalization test for project defense.
