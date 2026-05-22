# Step 09 — Evaluation and Model Comparison

## Goal

- Evaluate the baseline Logistic Regression model and the improved SNN using standard classification metrics (accuracy, confusion matrix, classification report) and print a final accuracy comparison.

## Files modified

- `src/evaluate.py`
- `src/snn_model.py`
- `main.py`

(`src/baseline_model.py` already returned `model`, `X_test`, `y_test`, `y_pred`, and `accuracy` — no change required.)

## What was implemented

- **`evaluate_classification(y_true, y_pred, model_name)`** in `src/evaluate.py`
  - Prints model name
  - Prints **accuracy**
  - Prints **confusion matrix** (`sklearn.metrics.confusion_matrix`)
  - Prints **classification report** (`sklearn.metrics.classification_report`)
  - Label names:
    - `0` = **Calm**
    - `1` = **Excited**

- **`train_snn_model`** now returns:
  - `model`, `X_test`, `y_test`, `y_pred`, `accuracy`
  - Test predictions computed on the same stratified 20% split (`random_state=42`) as the baseline

- **`main.py`**
  - After baseline training: `evaluate_classification(y_test, y_pred, "Baseline Logistic Regression")`
  - After SNN training: `evaluate_classification(snn_y_test, snn_y_pred, "Improved SNN")`
  - Final comparison:
    - `Baseline accuracy: ...`
    - `SNN accuracy: ...`

## Metrics used

- **Accuracy**
- **Confusion matrix** (2×2 for Calm vs Excited)
- **Classification report**: precision, recall, F1-score, support per class

## How to run

```bash
python main.py
```

## Expected output

After training messages and (for SNN) per-epoch loss lines, you should see two evaluation blocks, e.g.:

```
=== Baseline Logistic Regression ===
Accuracy: ...
Confusion matrix:
[[... ...]
 [... ...]]
Classification report:
              precision    recall  f1-score   support
        Calm       ...
     Excited       ...

=== Improved SNN ===
...

=== Comparison summary ===
Baseline accuracy: ...
SNN accuracy: ...
```

## Notes / limitations

- Feature extraction, binary arousal labeling, and model architectures are **unchanged**.
- Baseline and SNN each perform their own `train_test_split` with the same parameters; splits are reproducible but not guaranteed to be identical arrays unless you centralize splitting in a later step.
- No plots are saved yet (metrics are printed to the console only).
- Accuracy alone can be misleading under class imbalance; the classification report adds recall/F1 per class.
