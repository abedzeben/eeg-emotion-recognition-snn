# Final Presentation Summary — SEED Emotion Recognition

Generated: 2026-06-13 14:53 UTC

This report summarizes the **final presentation model**: CNN-SNN on the SEED dataset with subject-independent evaluation and `per_subject_per_channel` normalization (Step 38).

Reference target: **69.07% accuracy / 0.6937 Macro F1**

---

## Dataset

**SEED** (SJTU Emotion EEG Dataset)

| Property | Value |
|----------|-------|
| Number of subjects | 15 |
| Number of samples | 50,910 |
| Number of classes | 3 (Negative, Neutral, Positive) |
| Train subjects | [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11] |
| Test subjects | [12, 13, 14] |
| Train samples | 40,728 |
| Test samples | 10,182 |

---

## Model

**CNN-SNN hybrid**

| Setting | Value |
|---------|-------|
| Normalization | `per_subject_per_channel` |
| Evaluation | Subject-independent (train subjects 0–11, test 12–14) |
| CNN-SNN steps | 10 |
| Learning rate | 0.001 |
| Dropout | 0.3 |
| Beta | 0.95 |
| Epochs (max) | 100 |
| Best epoch | 2 |

---

## Results

| Metric | Value |
|--------|-------|
| **Accuracy** | **0.6907** (69.07%) |
| **Macro F1** | **0.6937** |
| **Weighted F1** | **0.6957** |

### Per-class metrics

| Class | Precision | Recall | F1 |
|-------|-----------|--------|-----|
| Negative | 0.5571 | 0.8342 | 0.6681 |
| Neutral | 0.6844 | 0.5507 | 0.6103 |
| Positive | 0.9678 | 0.6855 | 0.8025 |

### Confusion matrix (test set)

| True \\ Pred | Negative | Neutral | Positive |
|---|---|---|---|
| Negative | 2803 | 535 | 22 |
| Neutral | 1430 | 1824 | 58 |
| Positive | 798 | 306 | 2406 |

Figures: `results/presentation/figures/`

---

## Comparison

| Model | Accuracy | Macro F1 |
|-------|----------|----------|
| Logistic Regression (baseline) | 0.5463 | 0.5263 |
| Previous CNN-SNN (global norm) | 0.4879 | 0.4912 |
| **Best CNN-SNN (this run)** | **0.6907** | **0.6937** |

### Improvement over baselines

| Comparison | Accuracy Δ | Macro F1 Δ |
|------------|------------|------------|
| vs Logistic Regression | +0.1444 (+26.4%) | +0.1674 (+31.8%) |
| vs Previous CNN-SNN | +0.2028 (+41.6%) | +0.2025 (+41.2%) |

---

## Conclusion

### Why SEED outperformed DEAP

SEED uses **cleaner 3-class emotion labels** (Negative / Neutral / Positive) derived from film clips, while DEAP maps continuous Valence–Arousal ratings into **four overlapping quadrants**. Binary validation on DEAP (Step 43) showed the Temporal SNN learns signal well (~71% Valence, ~70% Arousal) but **4-class VA mapping caps performance around 53%**. SEED's label clarity and balanced task definition allow the same modeling family to reach **~69% accuracy**.

### Why `per_subject_per_channel` normalization helped

EEG amplitude varies strongly across subjects and electrode sites. **Global normalization** mixes statistics across subjects and washes out subject-specific patterns. **Per-subject per-channel** normalization (Step 37) scales each channel using statistics from that subject's training trials only, reducing **inter-subject distribution shift** while preserving spatial structure for the CNN front-end. This improved Macro F1 from **0.4912 to 0.6937** over the previous global CNN-SNN.

### Why CNN-SNN succeeded on SEED

The **CNN** extracts spatial patterns across frequency bands and channels from SEED's `(5 x 62)` maps. The **SNN** temporal head integrates spike-based dynamics over multiple time steps, matching the project's neuromorphic focus. On SEED, this hybrid outperformed both logistic regression and a plain strong SNN, especially after subject-aware normalization.

### Why this model was selected for final presentation

This configuration is the **best reproducible result** on the held-out subject split: **69.07% accuracy / 0.6937 Macro F1**, beating all DEAP experiments and all prior SEED runs. It demonstrates a complete, validated pipeline (preprocessing, normalization, CNN-SNN, subject-independent metrics) suitable for academic presentation.

---

*Metrics exported to `results/presentation/`. DEAP research archived under `src/archive/deap_research/`.*
