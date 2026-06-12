# Step 33 — Research-Based SNN Accuracy Improvement

## Goal

Improve Temporal SNN **Accuracy** and **Macro F1** on DEAP multi-emotion classification without replacing the SNN or redesigning the pipeline.

## Baseline (Step 29)

| Metric | Value |
|--------|-------|
| Model | Temporal Windowed SNN (best fixed config) |
| Features | 40-channel windowed DE, 10 windows |
| Accuracy | **53.12%** |
| Macro F1 | **0.5103** |

## What was tested

Research-supported preprocessing changes that may help valence/arousal separation:

| Change | Rationale |
|--------|-----------|
| **EEG-only channels** | Remove peripheral sensors (EOG, EMG, GSR, etc.) that add noise unrelated to cortical emotion signals |
| **Subject-aware normalization** | Reduce inter-subject distribution shift (common in EEG emotion recognition) |
| **Log PSD features** | Complement DE with band-power information used widely in EEG affect studies |
| **DE + log PSD** | Richer spectral representation per temporal window |

### Configuration flags

```python
RUN_SNN_RESEARCH_EXPERIMENTS = True
SNN_USE_EEG_ONLY_CHANNELS = True
USE_BEST_TEMPORAL_SNN_CONFIG = True
TEMPORAL_SPIKE_ENCODING = False
RUN_CLASSICAL_MODELS = False
USE_FRONTAL_ASYMMETRY_FEATURES = False  # disabled for clean Step 33 comparison
MULTI_LABEL_STRATEGY = "mean"
```

Single-run flags (when `RUN_SNN_RESEARCH_EXPERIMENTS = False`):

```python
NORMALIZATION_MODE = "global"           # global | per_subject | per_subject_per_channel
TEMPORAL_FEATURE_TYPE = "de"            # de | log_psd | de_log_psd
SNN_USE_EEG_ONLY_CHANNELS = True
```

### Experiment grid

When `RUN_SNN_RESEARCH_EXPERIMENTS = True`, the pipeline runs **9 experiments**:

- **Normalization:** `global`, `per_subject`, `per_subject_per_channel`
- **Feature type:** `de`, `log_psd`, `de_log_psd`
- **Channels:** 32 EEG only → **160** features/window (`de`, `log_psd`) or **320** (`de_log_psd`)
- **Windows:** 10 (unchanged)
- **SNN:** fixed best config (hidden=128, second_hidden=32, beta=0.95, dropout=0.2, lr=0.0005, epochs=50)

Excluded from this step: LR, RF, binary task, grid search, temporal spike encoding.

## Implementation

| File | Changes |
|------|---------|
| `src/channel_selection.py` | `select_eeg_only_channels()`, `EEG_CHANNEL_NAMES` |
| `src/preprocessing.py` | `normalize_with_mode()`, `NORMALIZATION_MODES` |
| `src/features.py` | `extract_temporal_window_log_psd_features()`, `extract_temporal_features_by_type()` |
| `src/evaluate.py` | `evaluate_snn_research_experiment()`, `print_snn_research_summary()` |
| `src/snn_model.py` | `quiet` flag for research loops |
| `main.py` | Step 33 flags and `_run_snn_research_experiments()` |

### Normalization modes

| Mode | Description |
|------|-------------|
| `global` | Per trial and channel along time (legacy) |
| `per_subject` | One z-score per subject (stats over channels, trials, time) |
| `per_subject_per_channel` | Z-score per subject and channel (stats over trials and time) |

## Results (FAST_TEST_MODE, 8 subjects)

Run: `python main.py` with `FAST_TEST_MODE = True`, `MAX_SUBJECTS = 8`.

```
=== SNN Research Summary ===
normalization | feature_type | accuracy | macro_f1
per_subject_per_channel | de_log_psd | 0.5000 | 0.4616
global | log_psd | 0.5469 | 0.4600
per_subject_per_channel | de | 0.4844 | 0.4087
global | de_log_psd | 0.4844 | 0.4056
per_subject_per_channel | log_psd | 0.4531 | 0.3461
per_subject | log_psd | 0.4688 | 0.3380
global | de | 0.4375 | 0.3224
per_subject | de | 0.4531 | 0.2842
per_subject | de_log_psd | 0.4375 | 0.2762
```

### Best configuration (8-subject fast run)

| Setting | Value |
|---------|-------|
| Normalization | `per_subject_per_channel` |
| Feature type | `de_log_psd` |
| Shape | `(320, 10, 320)` |
| Accuracy | 50.00% |
| Macro F1 | **0.4616** |

### Comparison vs Step 29 baseline

On the **8-subject fast subset**, the best Macro F1 (**0.4616**) did **not** exceed the full-dataset baseline (**0.5103**).

Notable observations:

- **EEG-only + log PSD** (`global | log_psd`) reached the highest **accuracy** (54.69%) on the fast run, but Macro F1 remained below baseline.
- **Per-subject normalization** (pooled or per-channel) often **hurt** Macro F1 vs `global` on this subset.
- **DE + log PSD** gave the best Macro F1 among subject-per-channel setups, suggesting combined spectral features may help when subject shift is reduced.

## Full DEAP evaluation

For a fair comparison to the **53.12% / 0.5103** baseline, run:

```python
FAST_TEST_MODE = False
RUN_SNN_RESEARCH_EXPERIMENTS = True
```

```bash
python main.py
```

Review the `=== SNN Research Summary ===` table and baseline delta at the end.

## Conclusion (fast run)

Step 33 tests whether research-backed preprocessing can lift Temporal SNN performance. On 8 subjects:

- No configuration beat the published full-dataset baseline on **Macro F1**.
- **Log PSD** and **DE + log PSD** are promising directions; **per_subject_per_channel + de_log_psd** ranked first among the 9 configs.
- A full 32-subject run is required to determine if any setup exceeds **53.12% accuracy** and **0.5103 Macro F1**.

## How to use single best setup

After research, disable the grid and apply the winner:

```python
RUN_SNN_RESEARCH_EXPERIMENTS = False
NORMALIZATION_MODE = "per_subject_per_channel"  # or best from full run
TEMPORAL_FEATURE_TYPE = "de_log_psd"
SNN_USE_EEG_ONLY_CHANNELS = True
```
