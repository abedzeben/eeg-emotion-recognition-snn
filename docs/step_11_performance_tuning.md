# Step 11 — Performance Tuning and Balanced Metrics

## Goal

- Tune baseline and SNN models to achieve **balanced** performance across Calm and Excited classes—not only high accuracy on the majority class.

## Problem discovered in previous steps

- **Step 9:** Both models often reached high accuracy while predicting mostly **Calm** (majority class).
- **Step 10:** Class balancing improved **Excited** detection, but overall accuracy sometimes dropped.

This step adds **scaling** and **hyperparameter search**, selecting models by **macro F1** instead of accuracy alone.

## Why accuracy alone is not enough

With imbalanced labels, a model can score high accuracy by always predicting the common class. **Macro F1** averages F1 across classes (Calm and Excited) with equal weight, rewarding models that perform reasonably on **both** classes.

## What tuning was added

### Baseline (`train_baseline_model`)

- **Pipeline:** `StandardScaler()` → `LogisticRegression()`
- **Grid search** (8 configurations):
  - `class_weight`: `None`, `'balanced'`
  - `C`: `0.01`, `0.1`, `1`, `10`
- Same split: `test_size=0.2`, `random_state=42`, `stratify=y`
- Per config: accuracy, precision, recall, F1, macro F1 (logged)
- **Best model:** highest **macro F1**

### SNN (`train_snn_model`)

- **StandardScaler** on features (fit on train, apply to test)
- Architecture unchanged in structure: `Input → Linear(hidden) → LIF → Linear(32) → LIF → Linear(32, 2)`
- **Grid search** (16 configurations):
  - `hidden_size`: `32`, `64`
  - `learning_rate`: `0.001`, `0.0005`
  - `epochs`: `50`, `100`
  - `class_weight`: `None`, `'balanced'` (weighted `CrossEntropyLoss` when balanced)
- **Best model:** highest **macro F1**
- Epoch loss prints are suppressed during the search to keep runtime manageable; each config prints one summary line.

### Evaluation (`evaluate_classification`)

- Still prints accuracy, confusion matrix, classification report
- Also prints **Macro F1**

## Metrics used

- Accuracy
- Macro precision / recall / F1 (during tuning)
- **Macro F1** for model selection
- Per-class metrics in `classification_report` (Calm / Excited)

## How best model is selected

The configuration with the **highest macro F1** on the held-out test split is kept. Accuracy is reported but not used as the primary selection criterion.

## How to run

```bash
python main.py
```

## Expected output

- One line per baseline configuration tried, then `Selected baseline params: ...`
- One line per SNN configuration tried, then `Selected SNN params: ...`
- Full `evaluate_classification` blocks for baseline and SNN (including Macro F1)
- Final comparison:

```
=== Comparison summary ===
Best Baseline Accuracy: ...
Best Baseline Macro F1: ...
Best Baseline Params: {...}
Best SNN Accuracy: ...
Best SNN Macro F1: ...
Best SNN Params: {...}
```

## Notes / limitations

- Feature extraction, DEAP loading, and binary arousal labels are **unchanged**.
- SNN tuning runs **16** trainings; with many DEAP subjects this can take a long time—reduce subject files for faster experiments.
- Baseline and SNN each use an independent `train_test_split` with the same parameters (reproducible, but not a single shared test index).
- Multi-emotion classification is not implemented.
- No plots saved to `results/` yet—metrics are console-only.
