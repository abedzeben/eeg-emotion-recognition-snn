# EEG Emotion Recognition (Baseline + SNN)

This repository contains a starter project structure for EEG-based emotion recognition, including preprocessing, feature extraction, a classical baseline model, and a spiking neural network (SNN) model.

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

Run the example pipeline (uses a small synthetic dataset by default):

```bash
python main.py
```

## Notes

- Put your real data under `data/raw/` and adapt `src/load_data.py` accordingly.
- The SNN model is a minimal template intended to be extended once your data/labels pipeline is stable.
