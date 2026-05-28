# Step 13 — Multi-Emotion Classification (Valence–Arousal)

## Goal

* Extend the project from **binary arousal** (Calm vs Excited) to **4-class emotion** classification using DEAP **Valence** and **Arousal** ratings.

## Why Valence + Arousal are used

DEAP provides continuous self-report scores. Valence (pleasant vs unpleasant) and Arousal (calm vs excited) form a 2D affect space.

Initially, fixed thresholds (**5**) were used.

However, this produced a missing **Happy / Excited** class (0 samples).

The project now uses **adaptive median thresholds** based on the actual dataset distribution.

## Thresholding strategy

Thresholds are computed dynamically:

```python
v_threshold = np.median(valence)
a_threshold = np.median(arousal)
```

Current values:

```text
Valence threshold: 3.08
Arousal threshold: 3.315
```

## Class mapping

| Class | Name             | Valence  | Arousal  |
| ----- | ---------------- | -------- | -------- |
| 0     | Calm / Relaxed   | > median | ≤ median |
| 1     | Happy / Excited  | > median | > median |
| 2     | Sad / Low        | ≤ median | ≤ median |
| 3     | Angry / Stressed | ≤ median | > median |

Implemented in `src/labels.py`.

## Resulting class distribution

```text
Class 0 (Calm / Relaxed): 253
Class 1 (Happy / Excited): 386
Class 2 (Sad / Low): 387
Class 3 (Angry / Stressed): 254
```

This resolved the previous zero-sample issue and enabled full 4-class training.

## Files modified

* `src/labels.py` — adaptive median threshold labeling
* `main.py` — multi-emotion pipeline; binary kept behind `RUN_BINARY_CLASSIFICATION`
* `src/baseline_model.py` — multi-class logistic regression support
* `src/snn_model.py` — dynamic `output_size = num_classes`
* `src/evaluate.py` — multi-class evaluation support

## How to run

```bash
python main.py
```

Optional:

```python
RUN_BINARY_CLASSIFICATION = True
```

to also run the legacy binary pipeline.

SNN mode:

* `USE_SPIKE_ENCODING = False` → tuned SNN (default)
* `USE_SPIKE_ENCODING = True` → spike-encoded SNN

## Expected output

* `Multi-emotion labels created`
* Balanced 4-class distribution
* Baseline evaluation (4×4 confusion matrix)
* Multi-emotion SNN evaluation
* Comparison summary between baseline and SNN

## Current results

### Multi-Emotion Baseline

```text
Accuracy: 38.28%
Macro F1: 0.362
```

### Multi-Emotion SNN

```text
Accuracy: 39.06%
Macro F1: 0.366
```

The tuned SNN slightly outperformed the baseline model in the 4-emotion task.

## Notes / limitations

* Binary pipeline remains available through `RUN_BINARY_CLASSIFICATION`.
* Multi-emotion classification is significantly harder than binary classification.
* Real emotions overlap and DEAP ratings are subjective.
* Grid search for multi-class SNN is slower than binary tuning.
* Dominance and liking columns from DEAP are not used in this step.
