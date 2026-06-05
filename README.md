# EEG Emotion Recognition with Spiking Neural Networks

A complete Python pipeline for **EEG-based emotion recognition** on the [DEAP](https://www.eecs.qmul.ac.uk/mmv/datasets/deap/) dataset. The project compares a tuned **Logistic Regression baseline** against a **Spiking Neural Network (SNN)** built with PyTorch and [snntorch](https://snntorch.readthedocs.io/), covering both **binary arousal** and **4-class Valence–Arousal** emotion classification.

---

## Project Overview

This repository implements an end-to-end research pipeline:

1. Load multi-subject DEAP `.dat` files  
2. Preprocess raw EEG signals  
3. Extract statistical and frequency-band features  
4. Train and tune baseline and SNN classifiers  
5. Evaluate with confusion matrices, classification reports, and macro F1  
6. Export figures and structured metrics for reporting  

The **primary task** is **4-class multi-emotion classification** using adaptive median thresholds on Valence and Arousal. **Binary arousal classification** (Calm vs Excited) is also supported for comparison.

---

## Dataset

**DEAP** (Database for Emotion Analysis using Physiological signals) contains EEG and peripheral recordings from 32 participants watching music videos. Each participant has **40 trials**.

| Item | Detail |
|------|--------|
| Format | Preprocessed `.dat` files (`s01.dat` … `s32.dat`) |
| EEG shape | `(trials, channels, samples)` → typically `(40, 40, 8064)` per subject |
| Labels | Valence, Arousal, Dominance, Liking (columns 0–3) |
| Location | Place files in `data/raw/` |

The pipeline loads all available `s*.dat` files, sorts them by filename, and concatenates trials across subjects.

---

## Processing Pipeline

```
data/raw/s*.dat
    ↓  load_all_deap_files()
EEG trials (N, 40, 8064)
    ↓  bandpass_filter() + normalize()
Preprocessed EEG
    ↓  extract_features()
Feature matrix (N, 240)
    ↓  label creation
Binary (2-class)  OR  Multi-emotion (4-class)
    ↓  train_baseline_model() + train_tuned_snn_model()
Evaluation + visualization + export
```

### Preprocessing (`src/preprocessing.py`)

- Butterworth **bandpass filter** (0.5–50 Hz, `fs=128`)
- Per-trial/channel **z-score normalization**

### Feature Extraction (`src/features.py`)

Per channel, **6 features** (240 total for 40 channels):

| Feature | Description |
|---------|-------------|
| Mean | Time-domain average |
| Variance | Signal spread |
| Theta power | 4–8 Hz (Welch PSD) |
| Alpha power | 8–13 Hz |
| Beta power | 13–30 Hz |
| Gamma power | 30–45 Hz |

### Labeling (`src/labels.py`)

**Binary arousal** (legacy):

- `0` = Calm — Arousal ≤ 5  
- `1` = Excited — Arousal > 5  

**Multi-emotion** (primary) — adaptive **median thresholds**:

```python
v_threshold = np.median(valence)
a_threshold = np.median(arousal)
```

| Class | Emotion | Valence | Arousal |
|-------|---------|---------|---------|
| 0 | Calm / Relaxed | > median | ≤ median |
| 1 | Happy / Excited | > median | > median |
| 2 | Sad / Low | ≤ median | ≤ median |
| 3 | Angry / Stressed | ≤ median | > median |

---

## Results

### Binary Classification (Calm vs Excited)

| Model | Accuracy | Macro F1 |
|-------|----------|----------|
| Baseline Logistic Regression | **0.750** | **0.602** |
| Tuned SNN | 0.742 | 0.596 |

The baseline slightly outperforms the SNN on binary arousal classification.

### Multi-Emotion Classification (4-class Valence–Arousal)

| Model | Accuracy | Macro F1 |
|-------|----------|----------|
| Baseline Logistic Regression | 0.383 | 0.362 |
| Tuned SNN | **0.402** | **0.380** |

The tuned SNN slightly outperforms the baseline on the harder 4-class task. Multi-emotion accuracy is lower than binary, which is expected given class overlap and subjective ratings.

---

## Visualization

After each run, confusion matrices and metric comparisons are saved to `results/figures/`:

| File | Description |
|------|-------------|
| `binary_baseline_cm.png` | Binary baseline confusion matrix |
| `binary_snn_cm.png` | Binary SNN confusion matrix |
| `multi_baseline_cm.png` | Multi-emotion baseline confusion matrix |
| `multi_snn_cm.png` | Multi-emotion SNN confusion matrix |
| `accuracy_comparison.png` | Baseline vs SNN accuracy (multi-emotion) |
| `macrof1_comparison.png` | Baseline vs SNN macro F1 (multi-emotion) |

See `docs/step_14_visualization.md` for details.

---

## Exported Results

Structured metrics are written to `results/metrics/`:

| File | Format | Contents |
|------|--------|----------|
| `results_summary.csv` | CSV | Task, model, accuracy, macro F1, params, notes |
| `results_summary.json` | JSON | Same fields with nested `best_params` |

Four experiments are exported: binary baseline, binary SNN, multi-emotion baseline, multi-emotion SNN.

See `docs/step_15_results_export.md` for details.

---

## Installation

**Requirements:** Python 3.9+, DEAP `.dat` files in `data/raw/`

```bash
git clone <repository-url>
cd egg-emotion-recognition-snn

python -m venv .venv

# Windows PowerShell
.\.venv\Scripts\Activate.ps1

# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

**Key dependencies:** `numpy`, `pandas`, `scipy`, `scikit-learn`, `matplotlib`, `seaborn`, `torch`, `snntorch`, `mne`

---

## Usage

1. Download DEAP preprocessed data and place subject files in `data/raw/`:

   ```
   data/raw/s01.dat
   data/raw/s02.dat
   ...
   ```

2. Run the full pipeline:

   ```bash
   python main.py
   ```

3. Check outputs:

   - Console: evaluation metrics, comparison summary  
   - `results/figures/` — plots  
   - `results/metrics/` — CSV/JSON summary  

### Configuration (`main.py`)

| Flag | Default | Description |
|------|---------|-------------|
| `USE_SPIKE_ENCODING` | `False` | `False` = tuned SNN (Step 11); `True` = spike-encoded SNN (Step 12, experimental) |
| `RUN_BINARY_CLASSIFICATION` | `False` | Legacy flag; binary pipeline runs for full evaluation and export |

---

## Repository Structure

```
egg-emotion-recognition-snn/
├── data/
│   └── raw/                 # DEAP s*.dat files
├── docs/                    # Step-by-step guides (step_01 … step_16)
├── results/
│   ├── figures/             # Confusion matrices, metric plots
│   └── metrics/             # results_summary.csv / .json
├── src/
│   ├── load_data.py         # DEAP loading (single + multi-subject)
│   ├── preprocessing.py     # Bandpass filter, normalization
│   ├── features.py          # Mean, variance, band power features
│   ├── labels.py            # Binary + multi-emotion label creation
│   ├── baseline_model.py    # Tuned Logistic Regression
│   ├── snn_model.py         # Tuned + spike-encoded SNN
│   ├── evaluate.py          # Classification metrics
│   ├── visualize.py         # Confusion matrix + bar plots
│   └── results_export.py    # CSV/JSON export
├── main.py                  # Pipeline entry point
├── requirements.txt
└── README.md
```

---

## Documentation

Step-by-step development notes are in `docs/`:

| Step | Topic |
|------|-------|
| 01 | Project setup |
| 02 | DEAP loading |
| 03 | Preprocessing |
| 04–08 | Features, baseline, SNN, tuning |
| 09–11 | Evaluation, class imbalance, performance tuning |
| 12–12.1 | Spike encoding, SNN mode switch |
| 13 | Multi-emotion classification |
| 14–15 | Visualization, results export |
| 16 | Documentation polish |

---

## Future Work

- **Subject-independent evaluation** — leave-one-subject-out or cross-subject generalization  
- **Richer features** — differential entropy, CSP, or deep temporal representations  
- **Temporal SNN input** — rate or temporal encoding directly from EEG segments  
- **Hyperparameter search** — extend SNN tuning for multi-class tasks  
- **Streamlit demo** — interactive inference in `app/`  
- **Additional emotions** — incorporate Dominance and Liking dimensions  
- **Results dashboard** — automated report generation from `results/metrics/`  

---

## License & Attribution

This project uses the DEAP dataset. Please cite the DEAP paper and comply with the dataset license when publishing results.

---

## Acknowledgments

Built as a structured EEG emotion recognition pipeline combining classical machine learning and spiking neural networks for comparative analysis on DEAP.
