# Step 30 — Ambiguous Valence-Arousal Sample Filtering

## Goal

- Improve **Temporal SNN** accuracy by training only on **clear** emotion samples and discarding ambiguous trials near the neutral boundary.

## Problem

DEAP Valence and Arousal ratings near the middle of the scale (e.g. 4–6) are noisy. Including them adds label ambiguity that can hurt SNN learning.

## Filtering rule

Keep a sample only when **both** dimensions are clearly low or high:

```text
(Valence <= LOW_THRESHOLD  OR  Valence >= HIGH_THRESHOLD)
AND
(Arousal <= LOW_THRESHOLD  OR  Arousal >= HIGH_THRESHOLD)
```

Default: `LOW_THRESHOLD = 4.0`, `HIGH_THRESHOLD = 6.0`.

Samples with Valence or Arousal strictly between 4 and 6 are **removed**.

## Fixed 4-class labels (kept samples only)

| Class | Emotion | Rule |
|-------|---------|------|
| 0 | Calm / Relaxed | V ≥ HIGH, A ≤ LOW |
| 1 | Happy / Excited | V ≥ HIGH, A ≥ HIGH |
| 2 | Sad / Low | V ≤ LOW, A ≤ LOW |
| 3 | Angry / Stressed | V ≤ LOW, A ≥ HIGH |

This replaces the adaptive `MULTI_LABEL_STRATEGY` thresholds when filtering is enabled.

## Files modified

- `src/labels.py` — `create_clear_multi_emotion_labels()`, filter summary helpers
- `main.py` — `USE_AMBIGUOUS_SAMPLE_FILTER`, `LOW_THRESHOLD`, `HIGH_THRESHOLD`

## Configuration

```python
USE_AMBIGUOUS_SAMPLE_FILTER = True
LOW_THRESHOLD = 4.0
HIGH_THRESHOLD = 6.0

USE_TEMPORAL_SNN_FEATURES = True
USE_BEST_TEMPORAL_SNN_CONFIG = True
RUN_CLASSICAL_MODELS = False
```

## Expected output

```
=== Ambiguous sample filtering (Step 30) ===
LOW_THRESHOLD: 4.0
HIGH_THRESHOLD: 6.0
Original number of samples: 320
Remaining number of samples: ...
Removed number of samples: ...
Class distribution after filtering:
  Class 0 (Calm / Relaxed): ...
  ...
WARNING: classes with fewer than 10 samples: ...   # if applicable
```

The same boolean mask is applied to:

- `X_features` (classical)
- `X_temporal_snn` (SNN)
- `y_multi`, `y_binary`, and raw `y`

## How to compare

| Run | `USE_AMBIGUOUS_SAMPLE_FILTER` |
|-----|-------------------------------|
| Before (Step 29) | `False` |
| After (Step 30) | `True` |

```bash
python main.py
```

Compare **Multi-Emotion Temporal SNN** accuracy and macro F1.

## Unchanged

- Temporal SNN architecture and best Step 28 config
- Train/test split inside `train_tuned_snn_model` (on filtered data)
- `evaluate_classification()` reporting
- Feature extraction pipeline (filtering runs after extraction)

## Notes

- With `FAST_TEST_MODE` and few subjects, some classes may drop below 10 samples — a warning is printed but training continues unless a class is empty.
- Classical models use the same filtered data when `RUN_CLASSICAL_MODELS = True`.
