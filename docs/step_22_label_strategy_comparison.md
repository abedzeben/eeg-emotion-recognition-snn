# Step 22 — Compare Label Threshold Strategies

## Goal

- Test whether different **Valence–Arousal threshold strategies** improve 4-class DEAP emotion classification (Accuracy / Macro F1) without removing the current **median** default.

## Strategies

| Strategy       | Valence threshold      | Arousal threshold      |
| -------------- | ---------------------- | ---------------------- |
| `median`       | `median(valence)`      | `median(arousal)`      |
| `mean`         | `mean(valence)`        | `mean(arousal)`        |
| `fixed_5`      | `5.0`                  | `5.0`                  |
| `quantile_60`  | 60th percentile        | 60th percentile        |
| `quantile_40`  | 40th percentile        | 40th percentile        |

Quadrant mapping (unchanged):

| Class | Emotion          | Valence   | Arousal   |
| ----- | ---------------- | --------- | --------- |
| 0     | Calm / Relaxed   | > t_v     | ≤ t_a     |
| 1     | Happy / Excited  | > t_v     | > t_a     |
| 2     | Sad / Low        | ≤ t_v     | ≤ t_a     |
| 3     | Angry / Stressed | ≤ t_v     | > t_a     |

## Files modified

- `src/labels.py` — strategy parameter, threshold helpers, comparison helper
- `main.py` — `MULTI_LABEL_STRATEGY` flag, strategy preview, conditional training

## Configuration

```python
MULTI_LABEL_STRATEGY = "median"   # default
```

Other options: `"mean"`, `"fixed_5"`, `"quantile_60"`, `"quantile_40"`

## Runtime behavior

1. **`compare_label_strategies(y)`** — prints thresholds and class distribution for **all** strategies  
2. **`create_multi_emotion_labels(y, strategy=MULTI_LABEL_STRATEGY)`** — builds labels for the selected strategy  
3. If any class has **0 samples**, prints a **WARNING** and **skips** multi-emotion training (LR, SNN, RF)  
4. Binary classification always runs  
5. Full multi-emotion pipeline runs **only** for the selected strategy (when valid)

## Expected output

```
=== Label strategy comparison ===

Strategy: median
Valence threshold: ...
Arousal threshold: ...
Class distribution:
  ...

Strategy: fixed_5
...
WARNING: empty classes for strategy 'fixed_5': [1] (['Happy / Excited'])

Selected label strategy: median
Valence threshold: ...
Arousal threshold: ...
```

Then normal multi-emotion training if no empty classes.

## How to compare strategies

1. Review the printed distributions for all strategies  
2. Set `MULTI_LABEL_STRATEGY` to a valid strategy (no empty classes)  
3. Run `python main.py`  
4. Compare Accuracy and Macro F1 in the final model comparison  

Repeat with different `MULTI_LABEL_STRATEGY` values to find the best thresholding approach.

## Notes

- **`fixed_5`** may produce empty classes (e.g. Happy/Excited) on DEAP — training is skipped automatically  
- Binary labels (`y[:, 1] > 5`) are unchanged  
- Feature extraction, model architectures, and train/test splits are unchanged  
- To compare metrics across strategies, run the pipeline once per valid strategy and record results manually or extend export in a future step
