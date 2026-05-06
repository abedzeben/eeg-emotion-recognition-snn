# Step 06 — Simple Spiking Neural Network (SNN)

## Goal

- Train a minimal Spiking Neural Network (SNN) classifier on the extracted EEG features and report accuracy.

## Files modified

- `src/snn_model.py`
- `main.py`

## What was implemented

- **`train_snn_model(X, y)`** in `src/snn_model.py`
  - Converts `X` and `y` to PyTorch tensors.
  - Splits data using `train_test_split(test_size=0.2, random_state=42, stratify=y)`.
  - Builds a simple snntorch network:
    - `Linear(input_size → 32)` → `snn.Leaky` → `Linear(32 → 2)`
  - Trains with:
    - `CrossEntropyLoss`
    - Adam optimizer
    - a small number of epochs (10)
  - Returns:
    - trained model
    - accuracy on the held-out test split

- **`main.py`**
  - After the baseline model, trains the SNN using:
    - `snn_model, snn_acc = train_snn_model(X_features, y_binary)`
  - Prints:
    - `SNN model trained`
    - `SNN accuracy: <float>`

## How to run

```bash
python main.py
```

## Expected output

Along with earlier step prints (load, preprocessing, features, labels, baseline), you should see:

- `SNN model trained`
- `SNN accuracy: <float>`

## Notes

- This is intentionally a **simple** SNN template designed to validate that snntorch training runs end-to-end.
- The input is a feature vector (not a spike train). The current implementation repeats the same feature vector for a small number of simulation steps to drive the LIF layer.
