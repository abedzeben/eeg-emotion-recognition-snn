# Step 20 — Feature Selection and Random Forest

## Goal

- Test whether the current feature set can improve **4-class multi-emotion** classification using **SelectKBest** feature selection and a tuned **Random Forest** baseline, without changing labels or the SNN.

## Files added / modified

- `src/random_forest_model.py` (new)
- `main.py` — trains RF after multi-emotion LR + SNN, prints final comparison

## What was implemented

### `train_random_forest_model(X, y)`

- Same split: `test_size=0.2`, `random_state=42`, `stratify=y`
- **Pipeline:** `SelectKBest(f_classif)` → `RandomForestClassifier`
- **Feature selection k:** 50, 100, 200, 300 (skips k > number of features)
- **Random Forest grid:**
  - `n_estimators`: 100, 200
  - `max_depth`: None, 10, 20
  - `class_weight`: None, `"balanced"`
- Prints **accuracy** and **macro F1** per configuration
- Selects best model by **macro F1**

### Unchanged

- Labels (`src/labels.py`)
- Logistic Regression baseline (`src/baseline_model.py`) — still uses `StandardScaler`
- SNN (`src/snn_model.py`) — still uses `StandardScaler` internally
- Evaluation, visualization, export pipelines

## Final comparison output

After multi-emotion training:

```
=== Final Model Comparison (Multi-Emotion) ===
Logistic Regression | Accuracy: ... | Macro F1: ...
SNN                 | Accuracy: ... | Macro F1: ...
Random Forest       | Accuracy: ... | Macro F1: ...
```

## How to run

```bash
python main.py
```

## Notes

- Random Forest does not use `StandardScaler` (tree models are scale-invariant); LR and SNN still scale features as before.
- With `USE_FREQUENCY_FEATURES = True`, input dimension is 440; all k values apply.
- With 240 features, `k=300` is skipped automatically.
- RF grid search adds runtime on top of existing LR and SNN tuning.
