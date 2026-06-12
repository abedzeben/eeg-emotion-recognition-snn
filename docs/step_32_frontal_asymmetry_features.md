# Step 32 — Frontal Asymmetry Features for Temporal SNN

## Goal

- Improve Temporal SNN emotion recognition by adding **frontal EEG asymmetry** features, which are widely used for valence and stress-related affect.

## Motivation

The Temporal SNN can confuse **Calm/Relaxed** vs **Angry/Stressed**. Frontal alpha/beta asymmetry (right − left) often correlates with emotional valence and arousal in DEAP-style studies.

## Feature design

Per temporal window, compute Differential Entropy per channel and band, then:

```text
asymmetry = DE_right − DE_left
```

| Channel pair | Bands |
|--------------|-------|
| F3 / F4 | delta, theta, alpha, beta, gamma |
| F7 / F8 | delta, theta, alpha, beta, gamma |
| Fp1 / Fp2 | delta, theta, alpha, beta, gamma |

**Asymmetry features per window:** 3 × 5 = **15**

| Component | Features / window |
|-----------|-------------------|
| Windowed DE (Step 27) | 200 |
| Frontal asymmetry (Step 32) | 15 |
| **Combined** | **215** |

Shape: `(num_trials, 10, 215)` with default 10 windows.

## Configuration

```python
USE_TEMPORAL_SNN_FEATURES = True
USE_FRONTAL_ASYMMETRY_FEATURES = True
TEMPORAL_NUM_WINDOWS = 10          # via features.TEMPORAL_NUM_WINDOWS
USE_BEST_TEMPORAL_SNN_CONFIG = True
TEMPORAL_SPIKE_ENCODING = False
RUN_CLASSICAL_MODELS = False
MULTI_LABEL_STRATEGY = "mean"
```

Set `USE_FRONTAL_ASYMMETRY_FEATURES = False` to reproduce Step 27/29 (200 features/window).

## Expected output

```
=== Frontal asymmetry features (Step 32) ===
Frontal asymmetry enabled
Number of asymmetry features per window: 15
Temporal SNN feature shape: (num_trials, 10, 215)
SNN input per time step: 215
Number of time steps: 10
```

## Files modified

- `src/features.py` — asymmetry extraction, `extract_temporal_window_snn_features()`
- `main.py` — `USE_FRONTAL_ASYMMETRY_FEATURES`

## Baseline to beat

| Metric | Step 29 (10 windows, 200 feats) |
|--------|----------------------------------|
| Accuracy | 53.12% |
| Macro F1 | 0.5103 |

## Unchanged

- Labels and `MULTI_LABEL_STRATEGY`
- Classical combined Stat+DE features (LR/RF when enabled)
- Best Temporal SNN hyperparameters
- Train/test split and evaluation

## How to compare

```bash
python main.py
```

Toggle `USE_FRONTAL_ASYMMETRY_FEATURES` and compare Multi-Emotion Temporal SNN metrics.
