# Step 21 — Remove Constant and Near-Constant Features

## Goal

- Improve multi-emotion (and binary) classification by removing **constant or near-constant** features before model training, reducing noise for SelectKBest and classifiers.

## Problem

During Random Forest feature selection (`SelectKBest`), sklearn may warn that many features are **constant**. Constant features carry no discriminative information and can hurt feature selection and model performance.

## Solution

`remove_constant_features(X, threshold=0.0)` in `src/features.py`:

- Uses `sklearn.feature_selection.VarianceThreshold`
- Drops columns with **variance ≤ threshold**
- Default `threshold=0.0` removes exactly constant features; increase slightly (e.g. `1e-8`) for near-constant features

## Files modified

- `src/features.py` — `remove_constant_features()`
- `main.py` — optional cleaning step before all training

## Flag

```python
REMOVE_CONSTANT_FEATURES = True   # default
```

Set to `False` to skip cleaning and use the raw feature matrix.

## When it runs

After feature extraction, **before** Logistic Regression, SNN, and Random Forest training. The same cleaned matrix is passed to all models.

## Expected output

```
Original feature shape: (num_trials, n_features)
Cleaned feature shape: (num_trials, n_features - n_removed)
Removed constant features: n_removed
```

## What was not changed

- Labels and median Valence–Arousal mapping  
- Train/test split inside each trainer (`random_state=42`, `stratify=y`)  
- SNN architecture and tuning grid  
- Logistic Regression and Random Forest training logic  
- Evaluation, visualization, and results export  

## How to run

```bash
python main.py
```

## Notes

- Cleaning is fit on the **full feature matrix** before per-model splits (consistent with prior pipeline design).
- If all features are non-constant, `n_removed` is 0 and shape is unchanged.
- Works with both 240-feature (statistical) and 440-feature (frequency) modes.
