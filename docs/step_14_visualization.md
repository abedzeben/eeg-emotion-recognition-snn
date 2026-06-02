# Step 14 — Visualization and Results Analysis

## Goal

- Add professional evaluation plots for binary and multi-emotion classification, saved under `results/figures/`.

## Files added / modified

- `src/visualize.py` (new)
- `main.py` — calls visualization after evaluation
- `requirements.txt` — added `seaborn`

## Functions

### `plot_confusion_matrix(cm, labels, title, save_path)`

- Matplotlib + seaborn heatmap
- Annotated cell values
- Axes: **Predicted** (x), **True** (y)
- Saves figure to disk

### `plot_metric_comparison(results_dict, metric_name, save_path)`

- Bar chart comparing **Baseline** vs **SNN**
- Used for Accuracy and Macro F1 (multi-emotion task)

### `generate_all_figures(...)`

- Creates `results/figures/` if missing
- Saves all Step 14 outputs in one call

## Output files

```
results/figures/
    binary_baseline_cm.png
    binary_snn_cm.png
    multi_baseline_cm.png
    multi_snn_cm.png
    accuracy_comparison.png
    macrof1_comparison.png
```

## How to run

```bash
python main.py
```

After training and evaluation, the pipeline prints:

```
Visualization completed
```

## Notes

- Model training logic is unchanged; only post-evaluation plotting was added.
- Binary and multi-emotion pipelines both run to produce all confusion matrices.
- Metric comparison plots use **multi-emotion** Baseline vs SNN results.
