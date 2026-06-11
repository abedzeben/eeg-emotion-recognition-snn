# Step 30 — Temporal Spike Encoding (NeuCube-Inspired)

## Goal

- Improve **Temporal SNN** performance by converting windowed DE features into **spike trains** before the network, following NeuCube-inspired SNN literature.

## Motivation

Step 27 showed that sequential windowed DE features outperform static repetition. NeuCube-style approaches further argue that SNNs benefit when temporal information is expressed as **spikes** rather than continuous values alone.

## Pipeline

```text
EEG
  → Windowed DE features (10 × 200)
  → Temporal rate spike encoding (10 windows × 10 encoding steps × 200)
  → Temporal SNN (best Step 28/29 config)
```

| Stage | Shape |
|-------|-------|
| Windowed DE | `(trials, 10, 200)` |
| After encoding | `(trials, 10, ENCODING_STEPS, 200)` |
| SNN batch input | `(batch, 10, 10, 200)` |

## Rate encoding

Per DE feature dimension (fit on **training** data only):

1. Min–max normalize each feature across all training trials and windows.
2. Treat normalized value as **spike probability** (higher DE → more spikes).
3. Sample binary spikes over `ENCODING_STEPS` (default 10) per window.

Encoding is applied **inside SNN training** (not stored as a separate dataset file).

## SNN forward pass (Step 30)

For each temporal window `w` and encoding step `e`:

```text
spike_vector = x[:, w, e, :]
→ Linear → LIF → Dropout → Linear → LIF → Dropout → Linear
```

Output is averaged over all `windows × encoding_steps` (100 steps by default).

## Configuration

```python
USE_TEMPORAL_SNN_FEATURES = True
TEMPORAL_SPIKE_ENCODING = True
ENCODING_STEPS = 10
USE_BEST_TEMPORAL_SNN_CONFIG = True
RUN_CLASSICAL_MODELS = False
```

Set `TEMPORAL_SPIKE_ENCODING = False` to reproduce Step 27/29 continuous temporal input.

## Expected output

```
=== Temporal spike encoding (Step 30) ===
Temporal spike encoding enabled
Encoding steps: 10
Temporal encoded shape: (num_trials, 10, 10, 200)
SNN input shape: (num_trials, 10, 10, 200)

Running Temporal Spike-Encoded SNN (Step 30)
Using best Temporal SNN config only
Skipping grid search
```

## Files modified

- `src/snn_model.py` — rate encoding helpers, 4D `SimpleSNN.forward(temporal_spike=True)`
- `main.py` — `TEMPORAL_SPIKE_ENCODING`, `ENCODING_STEPS`

## Unchanged

- Windowed DE feature extraction `(trials, 10, 200)`
- Labels, train/test split, evaluation
- Best Temporal SNN hyperparameters (hidden 128, second 32, lr 0.0005, etc.)
- Logistic Regression, Random Forest, classical features

## How to compare

| Mode | Flag |
|------|------|
| Continuous temporal (Step 29) | `TEMPORAL_SPIKE_ENCODING = False` |
| Temporal spikes (Step 30) | `TEMPORAL_SPIKE_ENCODING = True` |

```bash
python main.py
```

Compare Multi-Emotion Temporal SNN accuracy and macro F1.
