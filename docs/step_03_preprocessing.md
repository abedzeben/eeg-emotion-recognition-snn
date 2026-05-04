# Step 03 — EEG Preprocessing

## Goal

- Implement basic EEG preprocessing for DEAP-shaped data: bandpass filtering and per-trial/channel normalization.

## Files modified

- `src/preprocessing.py`
- `main.py`

## What was implemented

- `bandpass_filter(data, low=0.5, high=50, fs=128)` in `src/preprocessing.py`
  - Uses `scipy.signal.butter` and `scipy.signal.lfilter`
  - Filters along the **last axis** (time/samples)
  - Works with DEAP shape: `(trials, channels, samples)`
- `normalize(data)` in `src/preprocessing.py`
  - Normalizes each trial/channel signal using:
    - `(signal - mean) / (std + epsilon)`
  - Uses a small epsilon to avoid division by zero
- `main.py`
  - After loading `X, y`, runs:
    - `X_filtered = bandpass_filter(X)`
    - `X_normalized = normalize(X_filtered)`
  - Prints completion message and processed data shape
  - Keeps the existing DEAP file-not-found check

## How to run

```bash
python main.py
```

## Expected output

- If `data/raw/s01.dat` is missing:
  - `DEAP file not found. Please place s01.dat inside data/raw/`
- If present, you should see the DEAP inspection prints (from Step 02), followed by:
  - `Preprocessing completed`
  - `Processed data shape: (...)`

## Notes / limitations

- This step implements **only preprocessing** (no feature extraction, no models).
- `lfilter` is causal and may introduce phase delay; this is acceptable for this step’s inspection-focused workflow.
