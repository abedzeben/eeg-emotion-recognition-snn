# Step 10 — Class Imbalance Handling

## Goal

- Improve binary arousal classification (Calm vs Excited) by accounting for unequal class frequencies in both the baseline Logistic Regression model and the SNN trainer.

## Files modified

- `src/baseline_model.py`
- `src/snn_model.py`

(`main.py` evaluation and comparison logic is unchanged.)

## What was implemented

### Baseline (`train_baseline_model`)

`LogisticRegression` now uses:

```python
LogisticRegression(
    class_weight="balanced",
    random_state=42,
    max_iter=1000,
)
```

Sklearn automatically reweights classes inversely proportional to their frequency in the training set.

### SNN (`train_snn_model`)

Before training:

1. Compute class counts from `y_train`: `class_counts = np.bincount(y_train, minlength=2)`
2. Compute weights: `class_weights = len(y_train) / (2 * class_counts)`
3. Convert to a PyTorch tensor on the training device
4. Use weighted loss: `nn.CrossEntropyLoss(weight=class_weights_tensor)`

Architecture, epochs (50), Adam optimizer, and the rest of the training loop are unchanged.

## Why class imbalance matters

If one class (e.g. Calm or Excited) appears much more often than the other, a model can achieve high **accuracy** by mostly predicting the majority class while performing poorly on the minority class. Metrics such as recall and F1 for the rare class often improve when imbalance is addressed.

## How class weights work

- **Balanced logistic regression:** each class contributes equally to the loss in expectation, regardless of how many samples it has.
- **SNN weighted cross-entropy:** misclassifying a sample from a rare class incurs a **larger** loss than misclassifying a common class, encouraging the network to learn both classes.

Example (binary):

```text
class_weights[c] = n_train / (2 * count_c)
```

Higher weight → fewer training samples for that class.

## Expected output

Running `python main.py` should still show:

- Pipeline steps (load, preprocess, features, labels)
- Baseline and SNN training messages
- Per-epoch SNN loss lines
- `evaluate_classification` blocks for both models
- Comparison summary with baseline and SNN accuracy

Accuracies and per-class recall/F1 in the classification reports may **change** compared to unweighted training—often minority-class recall improves even if overall accuracy shifts slightly.

## Notes

- Feature extraction, binary arousal labels (`y[:, 1] > 5`), and SNN architecture are **not** modified.
- Class weights are computed on the **training split only** (same `train_test_split` as before).
- Extreme imbalance or very small datasets can still make learning difficult; consider more data or other strategies in future steps.
- Multi-emotion classification is not implemented in this step.
