# Step 07 — Improved SNN Architecture and Training

## Goal

- Improve the existing SNN implementation to achieve better performance on EEG emotion (binary arousal) classification, without changing feature extraction, labels, or the baseline model.

## Files modified

- `src/snn_model.py`
- `main.py`

## Architecture changes

The existing **`SimpleSNN`** class was extended (not replaced) with a deeper network:

**Before:**

- Input → Linear(32) → `snn.Leaky` → Linear(2)

**After:**

- Input → Linear(input_size, **64**) → `snn.Leaky`
- → Linear(**64**, **32**) → `snn.Leaky`
- → Linear(**32**, **2**)

Two LIF layers propagate spikes through two hidden stages before the binary output.

## Training changes

- Still uses **PyTorch**, **snntorch**, **`train_test_split`**, **CrossEntropyLoss**, and **Adam** (`lr=1e-3`).
- Training epochs increased from **10 → 50**.
- Each epoch prints average training loss, e.g.:
  - `Epoch 10 | Loss: 0.53`
- Evaluation accuracy is still computed on the held-out test split (20%, `random_state=42`, stratified).

## How to run

```bash
python main.py
```

## Expected output

After the usual pipeline prints (multi-subject load, preprocessing, features, labels, baseline), you should see:

- One line per epoch: `Epoch <n> | Loss: <value>`
- `Improved SNN model trained`
- `Improved SNN accuracy: <float>`

## Notes

- Feature extraction and binary arousal labeling (`y[:, 1] > 5`) are unchanged.
- The baseline Logistic Regression model is unchanged.
- Training on all DEAP subjects increases runtime and memory; fewer subject files will train faster.
- Accuracy can vary with random split and class balance; longer training (50 epochs) is intended to stabilize learning compared to the original 10-epoch setup.
