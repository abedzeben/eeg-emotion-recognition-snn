# Step 45 — DEAP Symmetric Difference Temporal SNN

## Goal

Test whether **EEG hemispheric asymmetry** (symmetric difference channels) improves DEAP 4-class Temporal SNN performance beyond the current best pooled result.

**Reference (Step 29):** 53.12% accuracy / 0.5103 Macro F1

---

## Motivation

High-accuracy DEAP literature often emphasizes **left–right asymmetry** rather than independent scalp channels. This step replaces 40-channel windowed DE with **14 symmetric difference channels** (left − right) while keeping the same Temporal SNN architecture.

---

## Symmetric difference channels

| # | Formula | # | Formula |
|---|---------|---|---------|
| 1 | Fp1 − Fp2 | 8 | T7 − T8 |
| 2 | AF3 − AF4 | 9 | CP5 − CP6 |
| 3 | F3 − F4 | 10 | CP1 − CP2 |
| 4 | F7 − F8 | 11 | P3 − P4 |
| 5 | FC5 − FC6 | 12 | P7 − P8 |
| 6 | FC1 − FC2 | 13 | PO3 − PO4 |
| 7 | C3 − C4 | 14 | O1 − O2 |

---

## Feature extraction

Same **Differential Entropy** pipeline as the temporal SNN, with literature-style bands on asymmetry channels:

| Band | Range (Hz) |
|------|------------|
| Theta | 4–8 |
| Alpha | 8–13 |
| Beta | 13–30 |
| Gamma | 30–45 |
| High Gamma | 45–63 |

**Note:** Baseline uses standard DE bands including **delta** (0.5–4 Hz). Asymmetry uses **5 bands without delta**, per Step 45 spec.

| Representation | Shape | Features / window |
|----------------|-------|-------------------|
| Baseline (40 ch) | `(1280, 10, 200)` | 40 × 5 DE |
| Symmetric diff (14 ch) | `(1280, 10, 70)` | 14 × 5 DE |

---

## Model & labels

Unchanged from Step 29:

| Parameter | Value |
|-----------|-------|
| hidden_size | 128 |
| second_hidden_size | 32 |
| beta | 0.95 |
| dropout | 0.2 |
| learning_rate | 0.0005 |
| epochs | 50 |
| class_weight | None |

- `MULTI_LABEL_STRATEGY = "mean"`
- No grid search, CNN, or spike encoding
- Full DEAP: **32 subjects, 1280 trials** (`FAST_TEST_MODE` ignored)

---

## Run

```python
RUN_DEAP_ASYMMETRY_SNN = True
```

```bash
python main.py
```

**Outputs:**

- `results/metrics/deap_asymmetry_snn_results.json`
- `results/metrics/deap_asymmetry_snn_results.csv`

---

## Dataset summary (full run)

| Item | Value |
|------|-------|
| Subjects | 32 |
| Trials | 1,280 |
| Class 0 (Calm) | 267 (20.86%) |
| Class 1 (Happy) | 330 (25.78%) |
| Class 2 (Sad) | 418 (32.66%) |
| Class 3 (Angry) | 265 (20.70%) |

---

## Results

### === Step 45 Comparison ===

| Model | Accuracy | Macro F1 | Weighted F1 |
|-------|----------|----------|---------------|
| **Baseline Temporal SNN** (this run, 10×200) | **45.70%** | **0.4345** | 0.4541 |
| **Symmetric Difference Temporal SNN** (10×70) | **41.02%** | **0.3782** | 0.3993 |
| Historical reference (Step 29) | 53.12% | 0.5103 | — |

### Deltas

| Comparison | Accuracy Δ | Macro F1 Δ |
|------------|------------|------------|
| Asymmetry vs baseline (same run) | **−4.69 pp** | **−0.0563** |
| Asymmetry vs Step 29 reference | **−12.10 pp** | **−0.1321** |

### Success criterion

**No improvement.** Symmetric difference features **did not** beat 53.12% / 0.5103.

**Conclusion:** *DEAP performance limitation is likely dominated by label ambiguity rather than feature representation.*

Asymmetry features should **not** replace the original 40-channel representation for this pipeline.

---

## Baseline run vs historical reference

This run’s baseline re-training yielded **45.70%** (below the documented **53.12%**). Both experiments in Step 45 share the same split and preprocessing, so the **asymmetry vs baseline comparison within this run is fair**. The gap to 53.12% may reflect training stochasticity (dropout, batch order) or minor pipeline differences vs the original Step 29 logged run. The asymmetry model still underperformed the **same-run** baseline.

---

## ANOVA: top 10 asymmetry channels

Mean / std / variance computed on flattened DE features; **mean ANOVA F-score** vs 4-class labels.

| Rank | Pair | Mean | Std | Variance | Mean F |
|------|------|------|-----|----------|--------|
| 1 | **PO3−PO4** | 2.56 | 1.27 | 1.61 | **10.46** |
| 2 | F3−F4 | 2.75 | 1.33 | 1.77 | 7.99 |
| 3 | P7−P8 | 3.00 | 1.28 | 1.65 | 6.65 |
| 4 | CP1−CP2 | 3.05 | 1.27 | 1.62 | 5.51 |
| 5 | AF3−AF4 | 2.77 | 1.35 | 1.81 | 5.13 |
| 6 | T7−T8 | 3.22 | 1.19 | 1.42 | 4.76 |
| 7 | O1−O2 | 2.79 | 1.24 | 1.54 | 3.94 |
| 8 | C3−C4 | 2.81 | 1.30 | 1.70 | 3.36 |
| 9 | FC5−FC6 | 2.92 | 1.33 | 1.78 | 3.11 |
| 10 | P3−P4 | 2.87 | 1.32 | 1.74 | 3.00 |

**Occipital (PO3−PO4)** and **frontal (F3−F4)** pairs show the strongest label-related variance — consistent with valence/arousal literature — but this signal **did not translate** into higher 4-class SNN accuracy when used alone.

---

## Files

| File | Role |
|------|------|
| `src/features.py` | Symmetric difference + windowed DE extraction |
| `src/deap_asymmetry_snn.py` | Step 45 experiment runner |
| `main.py` | `RUN_DEAP_ASYMMETRY_SNN` flag |

---

## Related steps

- Step 32 frontal asymmetry (additive 15 features): did not beat baseline
- Step 43 binary validation: SNN learns DEAP at ~70% binary
- Step 41 failure analysis: label ambiguity as main DEAP limiter

---

*Step 45 complete — asymmetry representation tested; no improvement over baseline or historical reference.*
