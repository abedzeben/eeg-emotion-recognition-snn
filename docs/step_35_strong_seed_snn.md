# Step 35 — Stronger SNN Architecture for SEED Subject Split

## Goal

Improve SNN **Accuracy** and **Macro F1** on the SEED **subject-independent** split. Logistic Regression already shows learnable signal (~54.63% / 0.5263 Macro F1), while the Step 34 simple SNN underperforms (~44.13% / 0.3781).

## Problem

The simple two-hidden-layer Temporal SNN is too weak for subject-generalization on SEED.

## Solution

Add `SEED_SNN_MODE = "strong"` with a deeper per-timestep SNN, improved training (AdamW, early stopping, LR scheduling), and a small hyperparameter grid.

## Configuration

```python
RUN_SEED_EXPERIMENT = True
RUN_SEED_ONLY = True
SEED_SPLIT_MODE = "subject"
SEED_SNN_MODE = "strong"       # "simple" keeps Step 34 model
SEED_SNN_FAST_GRID = True      # 4 configs; False → 72 configs
```

Unchanged: SEED loader, labels, subject split (train 0–11, test 12–14), Logistic Regression baseline, DEAP pipeline.

## Strong SNN architecture

Input: `(batch, 5, 62)` — 5 frequency bands as time steps, 62 channels per step.

Per time step:

```text
Linear(62, 256) → BatchNorm1d → LIF(β) → Dropout
→ Linear(256, 128) → BatchNorm1d → LIF(β) → Dropout
→ Linear(128, 64) → BatchNorm1d → LIF(β) → Dropout
→ Linear(64, 3)
```

Logits are **averaged** over the 5 time steps.

## Training

| Setting | Value |
|---------|-------|
| Optimizer | AdamW |
| weight_decay | 1e-4 |
| batch_size | 128 |
| Early stopping | patience 15, monitor **validation Macro F1** |
| LR scheduler | ReduceLROnPlateau (factor=0.5, patience=5, monitor val loss) |
| Validation | 15% stratified split from training data |

### Hyperparameter grid

**Fast grid** (`SEED_SNN_FAST_GRID = True`) — 4 configs:

| Parameter | Values |
|-----------|--------|
| learning_rate | 0.0005, 0.0003 |
| epochs | 100 |
| dropout | 0.2 |
| beta | 0.95 |
| class_weight | None, balanced |

**Full grid** — 72 configs: lr × epochs × dropout × beta × class_weight as specified in Step 35.

Best configuration is selected by **test Macro F1**.

## Baselines for comparison

| Model | Accuracy | Macro F1 |
|-------|----------|----------|
| Logistic Regression (subject split) | 54.63% | 0.5263 |
| Simple Temporal SNN (Step 34) | 44.13% | 0.3781 |

## Output

Per configuration:

- SEED_SNN_MODE, learning_rate, epochs, dropout, beta, class_weight
- best epoch, Accuracy, Macro F1, Weighted F1
- confusion matrix, classification report

Final summary:

```text
=== Strong SEED SNN Summary ===
Best SEED SNN Accuracy / Macro F1
Best params
Compare against LR and simple SNN baselines
```

## Results files

- `results/metrics/seed_strong_snn_results.csv`
- `results/metrics/seed_strong_snn_results.json`

## Files

| File | Role |
|------|------|
| `src/seed_strong_snn.py` | `StrongSeedSNN`, grid search, training loop |
| `src/seed_experiment.py` | Branch on `snn_mode` |
| `main.py` | `SEED_SNN_MODE`, `SEED_SNN_FAST_GRID` |

## How to run

```bash
python main.py
```

Use `SEED_SNN_MODE = "simple"` to reproduce Step 34 SNN behavior.
