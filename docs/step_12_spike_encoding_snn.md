# Step 12 — Rate-Based Spike Encoding for the SNN

## Goal

- Improve the SNN by converting EEG feature vectors into **spike trains** using rate encoding, so the network receives **temporal spiking input** instead of repeated static features.

## Why spike encoding was added

Previously, the SNN either fed feature vectors directly or repeated the same static input across time steps. That limits the temporal dynamics SNNs are designed for. Rate encoding provides a **stochastic, time-varying** input that better matches spiking neuron behavior.

## What rate encoding means

Each feature value is normalized to **[0, 1]** and interpreted as a **firing probability**. At each time step, a spike occurs at a given feature dimension with probability equal to that normalized value:

```text
spikes = rand() < normalized_feature
```

Over `num_steps` (default **10** for reasonable runtime), this produces a spike tensor of shape **`(num_steps, samples, features)`**.

## Files modified

- `src/snn_model.py`
- `main.py`

(Baseline model, labels, and feature extraction are unchanged.)

## Model changes

### `rate_encode_features(X_tensor, num_steps=10)`

- Input: `(samples, features)`
- Output: `(num_steps, samples, features)` spike tensor

### `SimpleSNN.forward(spikes)`

- Accepts spike input `(num_steps, batch, features)`
- Each time step: **Linear → LIF → Linear → LIF → Linear**
- Membrane states persist across steps; outputs are **accumulated and averaged** for the final prediction

### `train_snn_model`

- **StandardScaler** on features (train fit, test transform)
- Rate-encoded batches during training (new random spikes each batch)
- Fixed hyperparameters from Step 11 best starting point:
  - `hidden_size = 64`
  - `learning_rate = 0.0005`
  - `epochs = 100`
  - `class_weight = balanced` (weighted cross-entropy)
- Loss printed every **10** epochs
- Returns: `model`, `X_test`, `y_test`, `y_pred`, `accuracy`, `macro_f1`, `best_params`

## How to run

```bash
python main.py
```

## Expected output

- Baseline tuning and evaluation (unchanged)
- SNN training lines every 10 epochs: `Epoch 10 | Loss: ...`
- `Spike-encoded SNN model trained`
- `Spike-encoded SNN accuracy: ...`
- `Spike-encoded SNN macro F1: ...`
- `evaluate_classification` block for **Spike-encoded SNN** (accuracy, macro F1, confusion matrix, report)
- Final comparison summary with baseline and SNN metrics

## Notes / limitations

- `num_steps` is **10** (not 20) to keep runtime reasonable on multi-subject DEAP data.
- Test-set spikes are newly sampled at evaluation time (stochastic encoding).
- Step 11’s full SNN grid search was replaced by a **single** tuned configuration plus rate encoding; the baseline still uses its own Step 11 grid.
- Multi-emotion classification is not implemented.
- Performance may vary run-to-run due to random spike sampling.
