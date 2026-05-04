# Step 04 — Feature Extraction

## Goal

- Turn preprocessed EEG epochs into a fixed-size feature matrix per trial, and create binary arousal labels for downstream modeling (models are not implemented in this step).

## Files modified

- `src/features.py`
- `main.py`

## What was implemented

- **`extract_features(X)`** in `src/features.py`
  - Expects `X` with shape `(trials, channels, samples)`.
  - For each trial and each channel, computes **mean** and **variance** along the sample (time) axis.
  - Flattens to one vector per trial: channel order with pairs `[mean, var]` per channel, so `num_features = 2 * num_channels`.
  - Returns a matrix of shape `(num_trials, num_features)`.

- **`main.py`**
  - After preprocessing, calls `X_features = extract_features(X_normalized)` and prints completion plus `X_features` shape.
  - Builds **binary arousal** labels: `y_binary = (y[:, 1] > 5).astype(int)` (column 1 of DEAP labels = arousal), and prints their shape.

## How to run

```bash
python main.py
```

## Expected output

- If `data/raw/s01.dat` is missing: the same message as in earlier steps about placing `s01.dat` in `data/raw/`.
- Otherwise, in order: DEAP load inspection prints, preprocessing lines, then:
  - `Feature extraction completed`
  - `X_features shape: (num_trials, 2 * num_channels)`
  - `Labels created`
  - `y_binary shape: (num_trials,)`

## Notes

- This step does **not** train or evaluate models; it only produces `X_features` and `y_binary` for later steps.
- Arousal is taken from **label column 1** (0-based indexing), consistent with the DEAP label layout (valence, arousal, dominance, liking).
- If your label array has a different layout, adjust the column index for arousal accordingly.
