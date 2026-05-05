# Step 05 — Baseline Classification Model

## Goal

- Train a simple baseline classifier on extracted EEG features and report baseline accuracy (no SNN and no plots in this step).

## Files modified

- `src/baseline_model.py`
- `main.py`

## What was implemented

- **`train_baseline_model(X, y)`** in `src/baseline_model.py`
  - Splits data using `train_test_split(test_size=0.2, random_state=42, stratify=y)`.
  - Trains `LogisticRegression(max_iter=1000)`.
  - Returns:
    - trained model
    - `X_test`, `y_test`
    - `y_pred`
    - `accuracy`

- **`main.py`**
  - After feature extraction and binary arousal label creation, calls:
    - `model, X_test, y_test, y_pred, acc = train_baseline_model(X_features, y_binary)`
  - Prints:
    - `Baseline model trained`
    - `Baseline accuracy: <acc>`

## How to run

```bash
python main.py
```

## Expected output

Along with earlier step prints (load + preprocessing + features + labels), you should see:

- `Baseline model trained`
- `Baseline accuracy: <float>`

## Notes / limitations

- This is a **baseline** intended for quick validation of the pipeline, not a final model.
- No evaluation plots or detailed metrics are produced in this step.
- If the labels are highly imbalanced, accuracy may be misleading; more metrics can be added later.
