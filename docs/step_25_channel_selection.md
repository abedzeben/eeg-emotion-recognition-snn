# Step 25 — EEG Channel Selection for Emotion Recognition

## Goal

- Test whether **emotion-related EEG regions** (frontal and temporal) improve **4-class multi-emotion** classification compared to using all channels.

## Background

Emotion recognition literature often reports that **frontal** and **temporal** scalp regions carry strong valence–arousal information. This step subsets DEAP channels before feature extraction while keeping labels, models, and train/test splits unchanged.

## Channel selection modes

| Mode | Channels | Count |
|------|----------|-------|
| `all` | All 40 DEAP channels (32 EEG + 8 peripheral) | 40 |
| `frontal` | Fp1, Fp2, F3, F4, F7, F8, Fz | 7 |
| `frontal_temporal` | Frontal + T7, T8, FC5, FC6 | 11 |

DEAP channel order follows the standard 40-channel layout in `src/channel_selection.py`.

## Files modified

- `src/channel_selection.py` — channel names, modes, `select_channels()`
- `src/features.py` — `get_expected_feature_size()` scales with selected channel count
- `main.py` — `USE_CHANNEL_SELECTION`, `CHANNEL_SELECTION_MODE`

## Configuration

```python
USE_CHANNEL_SELECTION = True
CHANNEL_SELECTION_MODE = "all"          # baseline: all channels
# CHANNEL_SELECTION_MODE = "frontal"      # 7 channels
# CHANNEL_SELECTION_MODE = "frontal_temporal"  # 11 channels

USE_COMBINED_STAT_DE_FEATURES = True
MULTI_LABEL_STRATEGY = "mean"
```

## Expected output

```
=== Channel selection ===
Channel selection mode: frontal
Number of selected channels: 7
Selected channels: Fp1, Fp2, F3, F4, F7, F8, Fz
Data shape after channel selection: (num_trials, 7, 8064)

Feature type: Combined Statistical + Differential Entropy
Feature shape: (num_trials, 77)
Expected feature size: 77
```

Combined Stat+DE feature size = `11 × n_selected_channels` (e.g. 440 for 40 channels, 77 for 7 frontal channels).

## How to compare

Run three experiments with the **same** setup (Combined Stat+DE, mean labels, Random Forest):

```bash
python main.py
```

| Run | `CHANNEL_SELECTION_MODE` | Combined features |
|-----|--------------------------|-------------------|
| 1 | `all` | 440 |
| 2 | `frontal` | 77 |
| 3 | `frontal_temporal` | 121 |

Compare **Random Forest** accuracy and macro F1 in the comparison summary.

## Notes

- Channel selection runs **after** preprocessing and **before** feature extraction.
- Feature modes (statistical, frequency, DE, combined) are unchanged; only the channel dimension is reduced.
- Set `USE_CHANNEL_SELECTION = False` to skip selection (legacy behavior, no channel info printed).
- Peripheral channels (EOG, EMG, GSR, etc.) are included in `all` mode, matching prior pipeline behavior.
