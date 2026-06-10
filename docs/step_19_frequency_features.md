# Step 19 — Frequency-Domain EEG Features (Welch PSD)

## Goal

- Improve 4-class DEAP emotion classification by adding **frequency-domain band power** features alongside statistical descriptors, without changing labels or model architectures.

## Why frequency features are useful for EEG

Emotion-related brain activity often appears in specific frequency bands (e.g., alpha relaxation, beta arousal). **Power spectral density (PSD)** summarizes how signal energy is distributed across frequencies. Welch’s method provides a stable PSD estimate per channel, and **band power** captures rhythm-specific information that time-domain statistics alone may miss.

## Frequency bands used

| Band  | Range (Hz) | Sampling rate |
|-------|------------|---------------|
| Delta | 0.5–4      | 128 Hz        |
| Theta | 4–8        | 128 Hz        |
| Alpha | 8–13       | 128 Hz        |
| Beta  | 13–30      | 128 Hz        |
| Gamma | 30–45      | 128 Hz        |

Implementation: `scipy.signal.welch` → average PSD power inside each band per channel.

## Files modified

- `src/features.py` — optional frequency feature mode (legacy path preserved)
- `main.py` — `USE_FREQUENCY_FEATURES` flag and diagnostic prints

Training, labels, evaluation, visualization, and export are **unchanged**.

## Feature modes

### `USE_FREQUENCY_FEATURES = False` (default)

Legacy behavior — **statistical features only**:

- Per channel: **mean**, **std**, **variance**, **min**, **max**, **median**  
- No band-power features in legacy mode  
- DEAP: **240 features** `(40 × 6)`

### `USE_FREQUENCY_FEATURES = True` (Step 19)

- **Statistical (6 per channel):** mean, std, variance, min, max, median  
- **Band power (5 per channel):** delta, theta, alpha, beta, gamma (Welch)  
- DEAP: **440 features** = `240 statistical + 40 × 5 band powers`

## Expected feature size

| Mode   | Formula              | DEAP size |
|--------|----------------------|-----------|
| Legacy | 40 × 6               | **240**   |
| Step 19| 40 × 6 + 40 × 5      | **440**   |

## How to run

1. In `main.py`, set:

   ```python
   USE_FREQUENCY_FEATURES = True
   ```

2. Run:

   ```bash
   python main.py
   ```

Set `USE_FREQUENCY_FEATURES = False` for the 240-feature statistical-only pipeline.

## Expected output

When frequency mode is enabled:

```
Frequency feature extraction enabled
X_features shape: (num_trials, 440)
Expected feature size: 440
```

Then the existing pipeline continues: labels → baseline → SNN → figures → export.

## Notes

- **Labels, models, and SNN tuning grid are not modified** in this step — only input dimensionality changes when the flag is `True`.
- Models read `X.shape[1]` automatically; no architecture edits required.
- Re-run and update `results/metrics/` and README if you report new benchmark numbers.
- Legacy `extract_features()` default (`use_frequency_features=False`) uses **statistical features only** (240 dims on DEAP).
- Frequency mode adds Welch band powers on top of the same 6 statistics (440 dims on DEAP).
- Prior multi-emotion SNN baseline (~40.2% accuracy, ~0.380 macro F1) used 240 statistical features; Step 19 enables testing whether 440 features improve performance.
