# EEG Emotion Recognition (Baseline + SNN)

This repository implements an EEG-based emotion recognition pipeline on the DEAP dataset: preprocessing, frequency-band feature extraction, a tuned Logistic Regression baseline, and a tuned spiking neural network (SNN).

The **primary task** is **4-class multi-emotion classification** (Valence–Arousal quadrants) using **adaptive median thresholds** in `src/labels.py`. Legacy binary arousal classification (Calm vs Excited) remains available via `RUN_BINARY_CLASSIFICATION` in `main.py`.

## Project structure

- `data/raw/`: place raw EEG files here (e.g., `.edf`, `.fif`, `.set`, `.csv`)
- `notebooks/`: exploratory notebooks
- `src/`: reusable pipeline code (loading, preprocessing, features, models, evaluation)
- `results/figures/`: saved plots
- `results/metrics/`: saved metrics (e.g., `.json`, `.csv`)
- `app/`: Streamlit app code (optional)
- `docs/`: project documentation and step-by-step guides

## Setup

Create and activate a virtual environment, then install dependencies:

```bash
python -m venv .venv
# Windows PowerShell:
.\\.venv\\Scripts\\Activate.ps1
pip install -r requirements.txt
```

## Quick start

Place DEAP subject files (`s01.dat` … `s32.dat`) in `data/raw/`, then run:

```bash
python main.py
```

See `docs/step_13_multi_emotion_classification.md` for the current labeling strategy, class mapping, and latest results.

## Configuration (`main.py`)

- `USE_SPIKE_ENCODING = False` — tuned SNN (Step 11, default)
- `USE_SPIKE_ENCODING = True` — spike-encoded SNN (Step 12, experimental)
- `RUN_BINARY_CLASSIFICATION = False` — set `True` to also run legacy binary arousal classification

## Notes

- Multi-emotion labels use dataset medians: `v_threshold = median(valence)`, `a_threshold = median(arousal)` (not fixed score 5).
- Step-by-step guides live under `docs/` (`step_01` … `step_13`).
