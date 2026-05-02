# Step 01 — Project Setup

## Goal

- Establish a clean, reproducible project structure for an EEG emotion recognition pipeline (data → preprocessing → features → models → evaluation), with clear locations for notebooks, results, and documentation.

## Project structure

- `data/raw/`: raw EEG datasets (keep immutable copies here)
- `src/`: core Python modules (loading, preprocessing, features, models, evaluation)
- `notebooks/`: experiments and exploration
- `results/figures/`: generated plots
- `results/metrics/`: saved metrics/reports
- `app/`: app code (e.g., Streamlit UI)
- `docs/`: project documentation

## Files created

- `docs/step_01_project_setup.md`
- `src/baseline_model.py` (empty placeholder)
- `src/snn_model.py` (empty placeholder)
- `src/evaluate.py` (empty placeholder)

## How to run

1. Create and activate a virtual environment:

```bash
python -m venv .venv
# Windows PowerShell:
.\\.venv\\Scripts\\Activate.ps1
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run the project entry point:

```bash
python main.py
```

## Notes

- Put your dataset files under `data/raw/` and update your loading logic in `src/load_data.py` when you’re ready to use real data.
- Save plots to `results/figures/` and metrics (e.g., accuracy/F1/confusion matrix) to `results/metrics/` for consistent experiment tracking.
