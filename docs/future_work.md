# Future Work — DEAP Research Summary

This document preserves findings from DEAP experiments (Steps 39–45) and related analysis (Steps 41–42). All code remains available under `src/archive/deap_research/` with backward-compatible shims at `src/deap_*.py`. Nothing was deleted.

---

## DEAP Best Result (Reference)

| Metric | Value |
|--------|-------|
| Model | Temporal SNN (Step 29 config) |
| Features | 10 windows × 200 DE (40 channels × 5 bands) |
| Labels | `mean` Valence–Arousal quadrants (4-class) |
| Split | Random trial-level 80/20 |
| **Accuracy** | **53.12%** |
| **Macro F1** | **0.5103** |

Source: Steps 29–33, `results/metrics/`, `docs/step_41_deap_failure_analysis.md`

---

## Why DEAP Plateaued Around 53%

1. **Label ambiguity** — Continuous Valence/Arousal ratings mapped to four emotion quadrants create overlapping classes. Step 42 showed mild imbalance but not enough to explain the full ceiling.

2. **Task difficulty** — 4-class VA mapping is harder than binary or 3-class discrete labels. Step 43 binary validation reached **71.09% Valence** and **69.53% Arousal** with the same Temporal SNN pipeline, proving the model learns DEAP signal while the 4-class mapping limits accuracy.

3. **Temporal SNN was the main win** — Moving from static features (~45%) to windowed temporal DE features (+8%) was the largest DEAP improvement. Subsequent changes (spike encoding, asymmetry, CNN hybrid, subject normalization) did not beat 0.5103 Macro F1 on full DEAP.

4. **SEED contrast** — The same SNN family reached **69.07% / 0.6937** on SEED with cleaner 3-class labels and subject-independent evaluation, confirming the pipeline is sound.

See: `docs/step_41_deap_failure_analysis.md`, `docs/step_42_label_distribution_analysis.md`

---

## Step 39 — DEAP CNN-SNN (Failed to Beat Temporal Baseline)

**Code:** `src/archive/deap_research/deap_cnn_snn.py`  
**Docs:** `docs/step_39_deap_cnn_snn.md`

CNN-SNN with `per_subject_per_channel` normalization did not surpass the Temporal SNN baseline on DEAP. Spatial CNN features did not compensate for label noise and trial-level pooling.

**Flag:** `RUN_DEAP_CNN_SNN = True` (with `FINAL_MODE = "experimental"`)

---

## Step 40 — DEAP Temporal Normalization Study

**Code:** `src/archive/deap_research/deap_temporal_normalization_study.py`  
**Docs:** `docs/step_40_deap_temporal_normalization_study.md`

Compared normalization modes for Temporal SNN (no CNN). None beat the global-normalized Step 29 baseline on the evaluated subset.

**Flag:** `RUN_DEAP_TEMPORAL_NORMALIZATION_STUDY = True`

---

## Step 41–42 — Analysis Only

| Step | Document | Focus |
|------|----------|-------|
| 41 | `docs/step_41_deap_failure_analysis.md` | Full experiment post-mortem |
| 42 | `docs/step_42_label_distribution_analysis.md` | Label distributions across strategies |

No training code; metrics pulled from saved results.

---

## Step 43 — Binary Valence/Arousal Validation

**Code:** `src/archive/deap_research/deap_binary_validation.py`  
**Docs:** `docs/step_43_binary_validation.md`  
**Results:** `results/metrics/deap_binary_validation.json`

| Task | Accuracy | Macro F1 |
|------|----------|----------|
| Valence (binary) | 71.09% | 0.6146 |
| Arousal (binary) | 69.53% | 0.5705 |
| 4-class (reference) | 53.12% | 0.5103 |

**Conclusion:** SNN learns DEAP; 4-class VA mapping is the bottleneck.

**Flag:** `RUN_DEAP_BINARY_VALIDATION = True`

---

## Step 44 — Subject-Dependent Temporal SNN

**Code:** `src/archive/deap_research/deap_subject_dependent_snn.py`  
**Docs:** `docs/step_44_deap_subject_dependent_snn.md`  
**Results:** `results/metrics/deap_subject_dependent_snn_results.json`

Per-subject stratified CV (5-subject fast run): mean **41.50%** accuracy — worse than pooled baseline. Full 32-subject run not recommended.

**Flag:** `RUN_DEAP_SUBJECT_DEPENDENT_SNN = True`

---

## Step 45 — Symmetric Difference Asymmetry SNN

**Code:** `src/archive/deap_research/deap_asymmetry_snn.py`  
**Docs:** `docs/step_45_deap_asymmetry_snn.md`  
**Results:** `results/metrics/deap_asymmetry_snn_results.json`

Full DEAP: asymmetry features **41.02% / 0.3782** vs baseline **45.70% / 0.4345** in that run — no improvement.

**Flag:** `RUN_DEAP_ASYMMETRY_SNN = True`

---

## Subject Normalization Results (SEED Step 37)

On SEED, subject-aware normalization was critical:

| Normalization | Macro F1 (approx.) |
|---------------|-------------------|
| global (previous CNN-SNN) | 0.4912 |
| **per_subject_per_channel** | **0.6937** |

On DEAP (Step 40), subject normalization did not replicate this gain — dataset and label structure differ.

---

## Potential Future Improvements

### DEAP

1. **Binary or ordinal labels** — Use Valence/Arousal as separate binary tasks or regression instead of 4-class quadrants.
2. **Subject-independent evaluation** — Current DEAP best uses random trial split; subject-held-out evaluation may better reflect generalization (Step 44 explored per-subject training with poor pooled comparison).
3. **Label smoothing / soft targets** — Model continuous VA ratings directly rather than hard quadrant boundaries.
4. **Transfer from SEED** — Pre-train spatial CNN on SEED, fine-tune Temporal/SNN head on DEAP (not attempted).
5. **Larger temporal context** — Window count optimization (Step 31) found diminishing returns beyond 10 windows.

### SEED

1. **Cross-dataset validation** — Test SEED-trained models on DEAP or other EEG emotion datasets.
2. **Ensemble CNN-SNN configs** — Step 36 grid found a strong single config; ensembling may add marginal gains.
3. **Real-time deployment** — Spike encoding path for neuromorphic hardware (Step 30 experimental).

### General

1. **Attention over channels** — Learn channel weighting instead of fixed asymmetry pairs (Step 45).
2. **Mixed-frequency fusion** — Combine DE, log-PSD, and asymmetry with learned gating (Step 33 grid partially explored).

---

## How to Re-run Archived Experiments

```python
# main.py
RUN_PRESENTATION_MODE = False
FINAL_MODE = "experimental"
RUN_DEAP_BINARY_VALIDATION = True  # enable exactly ONE archive flag
```

List all archived experiments without training:

```python
RUN_PRESENTATION_MODE = False
FINAL_MODE = "experiment_archive"
```

Or import directly:

```python
from src.archive.deap_research import run_deap_binary_validation
```

---

## Related Files

| Type | Location |
|------|----------|
| Archive code | `src/archive/deap_research/` |
| Import shims | `src/deap_*.py` |
| DEAP results | `results/metrics/deap_*.json` |
| SEED best results | `results/metrics/seed_best_cnn_snn_results.json` |
| Presentation outputs | `results/presentation/` |
| Final presentation report | `docs/final_presentation_summary.md` (generated by presentation mode) |
