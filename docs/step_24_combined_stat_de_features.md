# Step 24 — Combined Statistical + Differential Entropy Features

## Goal

- Improve DEAP **4-class multi-emotion** classification by **concatenating** statistical and Differential Entropy (DE) features into one vector.

## Feature composition

| Component | Per channel | DEAP total |
|-----------|-------------|------------|
| Statistical (mean, std, variance, min, max, median) | 6 | **240** |
| Differential Entropy (delta, theta, alpha, beta, gamma) | 5 | **200** |
| **Combined** | 11 | **440** |

## Files modified

- `src/features.py` — `_extract_combined_stat_de_features()`, mode registry update
- `main.py` — `USE_COMBINED_STAT_DE_FEATURES` flag

## Configuration

```python
USE_COMBINED_STAT_DE_FEATURES = True
USE_DIFFERENTIAL_ENTROPY = False   # ignored when combined is True
USE_FREQUENCY_FEATURES = False
```

## Feature mode priority

1. **Combined Statistical + DE** (`USE_COMBINED_STAT_DE_FEATURES`)
2. Differential Entropy only (`USE_DIFFERENTIAL_ENTROPY`)
3. Frequency (`USE_FREQUENCY_FEATURES`)
4. Statistical (default)

## Expected output

```
=== Feature mode comparison ===
  Statistical Features: 240 features — ...
  Frequency Features: 440 features — ...
  Differential Entropy Features: 200 features — ...
  Combined Statistical + Differential Entropy (active): 440 features — ...

Feature type: Combined Statistical + Differential Entropy
Feature shape: (num_trials, 440)
Expected feature size: 440
```

## How to run

```bash
python main.py
```

Compare against:

| Mode | Features | Notes |
|------|----------|-------|
| DE only | 200 | Step 23 |
| Combined Stat+DE | 440 | Step 24 |
| Current best | — | ~42.58% multi-emotion accuracy |

Toggle flags and re-run with the same `MULTI_LABEL_STRATEGY = "mean"` for a fair comparison.

## Notes

- Labels, models (LR, SNN, RF), and train/test splits are **unchanged**.
- Combined mode reuses existing `_extract_features_legacy()` and `_extract_differential_entropy_features()` — no duplicate logic.
- Frequency mode (440) uses statistical + **Welch band power**; combined mode (440) uses statistical + **DE** — same size, different second half.
- Use `FAST_TEST_MODE` for quick experiments before full-dataset runs.
