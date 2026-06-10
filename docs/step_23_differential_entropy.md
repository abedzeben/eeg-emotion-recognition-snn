# Step 23 — Differential Entropy EEG Features

## Goal

- Improve DEAP **4-class multi-emotion** classification by adding **Differential Entropy (DE)** features computed from band-pass filtered EEG signals.

## Why Differential Entropy

DE is widely used in EEG emotion recognition. It measures the information content of a band-limited signal and often outperforms simple time-domain statistics for affect classification.

For a Gaussian signal with variance σ²:

```text
DE = 0.5 * log(2 * π * e * σ²)
```

## Frequency bands

| Band  | Range (Hz) | Sampling rate |
|-------|------------|---------------|
| Delta | 0.5–4      | 128 Hz        |
| Theta | 4–8        | 128 Hz        |
| Alpha | 8–13       | 128 Hz        |
| Beta  | 13–30      | 128 Hz        |
| Gamma | 30–45      | 128 Hz        |

Per channel: Butterworth band-pass filter → variance → DE.

## Files modified

- `src/features.py` — `_extract_differential_entropy_features()`, mode helpers
- `main.py` — `USE_DIFFERENTIAL_ENTROPY` flag

## Feature modes (comparison)

| Mode | Flag | DEAP size | Description |
|------|------|-----------|-------------|
| **Statistical** | default | 240 | mean, std, variance, min, max, median |
| **Frequency** | `USE_FREQUENCY_FEATURES = True` | 440 | statistical + Welch band power |
| **Differential Entropy** | `USE_DIFFERENTIAL_ENTROPY = True` | 200 | 5 DE values per channel |

Only **one** mode is active. Priority: DE > Frequency > Statistical.

## Configuration

```python
USE_DIFFERENTIAL_ENTROPY = True
USE_FREQUENCY_FEATURES = False   # ignored when DE is True
```

## Expected output

```
=== Feature mode comparison ===
  Statistical Features: 240 features — ...
  Frequency Features: 440 features — ...
  Differential Entropy Features (active): 200 features — ...

Feature type: Differential Entropy Features
Feature shape: (num_trials, 200)
Expected feature size: 200
```

## How to run

```bash
python main.py
```

Compare results against the current best (~42.58% multi-emotion accuracy) by toggling feature flags and re-running with the same `MULTI_LABEL_STRATEGY`.

## Notes

- Labels, models (LR, SNN, RF), and evaluation are **unchanged**.
- DE mode **replaces** the active feature vector (does not stack on statistical features).
- Statistical and frequency modes remain available when `USE_DIFFERENTIAL_ENTROPY = False`.
- Band-pass filtering adds runtime vs pure statistical features; use `FAST_TEST_MODE` for quick experiments.
- Reference benchmark to beat: **42.58% Accuracy** (prior best multi-emotion result).
