# Step 15 — Export Final Results and Metrics Summary

## Goal

- Save final evaluation metrics in structured files for reports, presentations, and GitHub documentation.

## Files modified

- `src/results_export.py` (new)
- `main.py` — builds results list and exports after visualization

## What was implemented

### `export_results_summary(results, output_dir="results/metrics")`

- Creates `output_dir` if missing
- Writes:
  - `results_summary.csv`
  - `results_summary.json`

Each record includes:

| Field        | Description                          |
| ------------ | ------------------------------------ |
| `task`       | Binary or Multi-Emotion classification |
| `model`      | Baseline Logistic Regression or Tuned SNN |
| `accuracy`   | Test accuracy                        |
| `macro_f1`   | Macro-averaged F1                    |
| `best_params`| Hyperparameters / config dict        |
| `notes`      | Short experiment description         |

### Experiments exported

1. Binary Classification — Baseline  
2. Binary Classification — Tuned SNN  
3. Multi-Emotion Classification — Baseline  
4. Multi-Emotion Classification — Tuned SNN  

## Output files

```
results/metrics/
    results_summary.csv
    results_summary.json
```

## Why exporting results is important

- **Reproducibility:** metrics and configs are stored outside console logs  
- **Reporting:** CSV/JSON are easy to import into slides, papers, or dashboards  
- **Version control:** track performance changes across commits or experiments  

## How to run

```bash
python main.py
```

## Expected output

After training, evaluation, and visualization:

```
Visualization completed
Results summary exported
Saved to results/metrics/results_summary.csv
Saved to results/metrics/results_summary.json
```

## Notes

- Model training, labeling, and visualization logic are unchanged.  
- `best_params` in CSV is stored as a JSON string; the `.json` file keeps nested dicts.  
- Multi-emotion labels use adaptive median thresholds (see Step 13).  
- SNN mode follows `USE_SPIKE_ENCODING` in `main.py` (tuned SNN by default).
