# Step 08 — Frequency Band Power Features

## Goal

- Improve EEG feature vectors by adding frequency-band power features (via Welch PSD) alongside mean and variance, to give baseline and SNN models richer inputs without changing labels or model architectures.

## Files modified

- `src/features.py`
- `main.py`

## What was implemented

- Extended **`extract_features(X)`** in `src/features.py` (existing function kept and extended).
- For each **trial** and **channel**:
  - **Mean** and **variance** along the time axis (unchanged).
  - **Band-average power** using `scipy.signal.welch` at **`fs = 128`**:
    - average PSD power inside each band (mean over frequency bins in the band).
- Per channel, **6 features** (in order):
  1. mean  
  2. variance  
  3. theta_power  
  4. alpha_power  
  5. beta_power  
  6. gamma_power  

- **`main.py`**
  - After feature extraction, prints:
    - `Improved feature extraction completed`
    - `X_features shape: ...`
  - Rest of pipeline unchanged (binary arousal labels, baseline, improved SNN).

## Frequency bands used

| Band  | Range (Hz) |
|-------|------------|
| Theta | 4–8        |
| Alpha | 8–13       |
| Beta  | 13–30      |
| Gamma | 30–45      |

Sampling frequency: **128 Hz**.

## Expected feature shape

- **Per trial:** `num_channels × 6` features.
- **DEAP (40 EEG channels):** `(num_trials, 240)` — e.g. `40 × 6 = 240`.

## How to run

```bash
python main.py
```

## Expected output

Among earlier pipeline messages, after preprocessing you should see:

- `Improved feature extraction completed`
- `X_features shape: (num_trials, 240)` when using standard DEAP channel count (40).

Then labels, baseline accuracy, SNN epoch losses, and improved SNN accuracy as in previous steps.

## Notes / limitations

- Band power is computed **per trial** with Welch; large multi-subject runs can be slower than mean/variance alone.
- Non-EEG channels in the 40-channel DEAP tensor are still included unless you filter channels in a later step.
- Labels, baseline model, and SNN architecture are **not** changed in this step—only feature dimensionality increases (e.g. from 80 to 240 for 40 channels).
