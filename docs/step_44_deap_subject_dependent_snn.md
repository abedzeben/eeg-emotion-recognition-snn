# Step 44 — DEAP Subject-Dependent 4-Class Temporal SNN

## Goal

Test whether DEAP 4-class SNN accuracy improves when using **subject-dependent** evaluation — training and testing separately per subject — as reported in much of the high-accuracy DEAP SNN literature.

**Not** subject-independent (unlike SEED Step 34+). Each subject gets their own model(s) via cross-validation on that subject's 40 trials only.

---

## Motivation

| Setting | Our result |
|---------|------------|
| **Pooled** (all 32 subjects, random 80/20 split) | **53.12%** accuracy / **0.5103** Macro F1 |

Literature often reports much higher DEAP accuracy under **subject-dependent** protocols (train/test within the same subject). This step tests whether our Temporal SNN can reach similar levels when evaluation matches that setting.

---

## Configuration

### Flags (`main.py`)

```python
RUN_DEAP_SUBJECT_DEPENDENT_SNN = True
SUBJECT_DEPENDENT_FAST_MODE = True          # False for full 32-subject run
MAX_SUBJECTS_FOR_SUBJECT_DEPENDENT = 5      # used when fast mode is True
MULTI_LABEL_STRATEGY = "mean"
```

When `RUN_DEAP_SUBJECT_DEPENDENT_SNN = True`, only Step 44 runs (no SEED, CNN-SNN, binary validation, or final comparison).

### Labels

- 4-class Valence–Arousal quadrants
- Strategy: **`mean`** — thresholds computed **per subject** on that subject's 40 trials (same rule as pooled pipeline, applied locally)

### Features

| Setting | Value |
|---------|-------|
| Type | Windowed Differential Entropy |
| Windows | 10 |
| Features / window | 200 (40 channels × 5 bands) |
| Preprocessing | Bandpass filter (same as main DEAP pipeline) |

### Model (Step 29 best config)

| Parameter | Value |
|-----------|-------|
| hidden_size | 128 |
| second_hidden_size | 32 |
| beta | 0.95 |
| dropout | 0.2 |
| learning_rate | 0.0005 |
| epochs | 50 |
| class_weight | None |

### Per-subject evaluation

For each subject:

1. Take 40 trials only.
2. Build 4-class labels (`mean` on subject ratings).
3. Print class distribution.
4. Skip if any class has **< 2** samples.
5. **Stratified 5-fold CV** (`shuffle=True`, `random_state=42`).
6. Each fold: train fresh SNN on 32 trials, test on 8 trials.

---

## Fast validation mode (default)

```python
SUBJECT_DEPENDENT_FAST_MODE = True
MAX_SUBJECTS_FOR_SUBJECT_DEPENDENT = 5
```

| Item | Value |
|------|-------|
| Subjects | 0, 1, 2, 3, 4 |
| Folds per subject | 5 |
| **Total SNN trainings** | **25** |

### Full run

```python
SUBJECT_DEPENDENT_FAST_MODE = False
```

| Item | Value |
|------|-------|
| Subjects | All 32 |
| **Total SNN trainings** | **160** |

---

## Run

```bash
python main.py
```

**Outputs:**

- `results/metrics/deap_subject_dependent_snn_results.json`
- `results/metrics/deap_subject_dependent_snn_results.csv`

---

## Fast-run results (5 subjects)

**Run date:** Step 44 initial fast validation  
**Subjects evaluated:** 5 | **Skipped:** 0

### Summary

| Metric | Subject-dependent (fast) | Pooled reference (Step 29) | Delta |
|--------|--------------------------|---------------------------|-------|
| **Mean accuracy** | **41.50%** | 53.12% | **−11.62 pp** |
| **Std accuracy** | 10.79% | — | — |
| **Mean Macro F1** | **0.2646** | 0.5103 | **−0.2457** |
| **Std Macro F1** | 0.0656 | — | — |
| **Best subject accuracy** | 52.50% (subject 2) | — | — |
| **Worst subject accuracy** | 22.50% (subject 4) | — | — |

### Per-subject results

| Subject | Class distribution (C/H/S/A) | Mean accuracy | Mean Macro F1 |
|---------|------------------------------|---------------|---------------|
| 0 | 8 / 7 / 17 / 8 | 50.00% | 0.3147 |
| 1 | 8 / 10 / 16 / 6 | 37.50% | 0.2756 |
| 2 | 12 / 5 / 9 / 14 | **52.50%** | **0.3527** |
| 3 | 6 / 10 / 20 / 4 | 45.00% | 0.1951 |
| 4 | 8 / 12 / 10 / 10 | 22.50% | 0.1851 |

*(C/H/S/A = Calm / Happy / Sad / Angry trial counts)*

### Fast-run recommendation

| Threshold | Result |
|-------------|--------|
| Accuracy > 70% | **No** (41.50%) |
| Macro F1 > 0.65 | **No** (0.2646) |

**Recommendation:** **Do not launch the full 32-subject run** based on this fast sample. Subject-dependent evaluation did **not** improve over pooled performance; it performed **substantially worse** on average.

---

## Interpretation

### Subject-dependent vs pooled

On the first 5 subjects, subject-dependent 4-class CV averaged **41.5%** accuracy vs **53.1%** pooled. High fold variance (e.g. subject 4: 12.5%–50% across folds) reflects:

- Only **32 training samples** per fold (vs 1,024 pooled)
- **8 test samples** per fold — high variance
- Per-subject **mean** thresholds on 40 trials — unstable quadrants
- Some subjects have **severely imbalanced** local classes (e.g. subject 3: 50% Sad)

### Comparison with Step 43 (binary)

Step 43 showed **~70%** binary accuracy with the **same SNN** on **pooled** data. Step 44 fast-run **~41%** on 4-class **per-subject** data. Together:

1. The SNN **can** learn DEAP affect signal (binary pooled ~70%).
2. **4-class quadrant labels** remain difficult even subject-dependently.
3. **Small per-subject sample size** (40 trials, 32 train / fold) limits subject-dependent 4-class learning.
4. Literature-high subject-dependent scores likely combine **easier tasks** (binary valence/arousal), **different features**, or **more training data per fold** — not reproduced here with 4-class VA quadrants on 40 trials.

### vs SEED

Do **not** compare these numbers to SEED subject-independent **69%** — different dataset, labels, and evaluation protocol.

---

## Full 32-subject run

**Status:** Not recommended after fast validation.

To run anyway (for completeness):

```python
SUBJECT_DEPENDENT_FAST_MODE = False
RUN_DEAP_SUBJECT_DEPENDENT_SNN = True
```

Expected runtime: ~160 SNN trainings (32 × 5 folds × 50 epochs each).

---

## Files

| File | Role |
|------|------|
| `src/deap_subject_dependent_snn.py` | Step 44 runner |
| `main.py` | Flags and exclusive run mode |
| `results/metrics/deap_subject_dependent_snn_results.json` | Full metrics |
| `results/metrics/deap_subject_dependent_snn_results.csv` | Per-subject summary |

---

## References

- Pooled baseline: `docs/step_41_deap_failure_analysis.md`
- Binary validation: `docs/step_43_binary_validation.md`
- Label strategy: `docs/step_42_label_distribution_analysis.md`
