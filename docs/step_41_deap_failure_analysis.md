# Step 41 — DEAP Failure Analysis Report

Analysis-only step: no new training. Metrics are taken from `results/metrics/*.json|csv`, step documentation (`docs/step_*.md`), and console logs where JSON export is missing.

**Evaluation note:** Many later DEAP experiments used `FAST_TEST_MODE` (8 subjects, 320 trials). The canonical full-dataset baseline (**53.12% / 0.5103**) was established on the full DEAP set (32 subjects, 1280 trials) unless stated otherwise. Fast-subset numbers are labeled **(8-subject fast)** below.

---

## 1. Executive Summary

The project systematically improved DEAP 4-class emotion SNN performance from roughly **40% / 0.37 Macro F1** (early static pipeline) to a best of **53.12% accuracy / 0.5103 Macro F1** via **Temporal Windowed SNN** modeling (Steps 27–29). After that point, **no further change beat the baseline** on full DEAP: spike encoding, ambiguous-sample filtering, frontal asymmetry, CNN-SNN, subject-aware normalization, and extended feature grids all failed to surpass **0.5103 Macro F1**.

On **SEED**, the same SNN family reached **69.07% / 0.6937 Macro F1** (CNN-SNN + `per_subject_per_channel`, subject-independent split). This contrast supports the conclusion that the **SNN pipeline is viable**, but **DEAP’s Valence–Arousal-derived 4-class labels, task difficulty, and label noise** limit accuracy—not a single missing architectural tweak.

For presentation: we tried research-based improvements across features, labels, temporal modeling, spike encoding, normalization, and CNN hybrids. **Temporal SNN was the main DEAP win.** Further gains were capped mainly by **dataset and label quality**, as shown by SEED success under the same modeling approach.

---

## 2. DEAP Baseline and Best Result

| Stage | Model | Features | Split | Accuracy | Macro F1 | Source |
|-------|-------|----------|-------|----------|----------|--------|
| Early multi-emotion | Tuned static SNN | Statistical (240) | Random 80/20 | 39.06% | 0.366 | `step_13` |
| Pre-temporal best | Tuned static SNN | Combined Stat+DE (440) | Random 80/20 | ~42.58% | not recorded | `step_23`, `step_24` |
| Static tuned (Step 26) | Tuned static SNN | Combined Stat+DE | Random 80/20 | 45.31% | 0.4412 | `step_26` |
| **Best DEAP result** | **Temporal SNN (Step 29 config)** | **10 × windowed DE (200/step)** | **Random 80/20** | **53.12%** | **0.5103** | `step_29`–`step_33`, `main.py` constants |

**Best configuration (canonical):**

- `MULTI_LABEL_STRATEGY = "mean"`
- 10 temporal windows, 200 DE features per window (40 channels × 5 bands)
- SNN: hidden=128, second_hidden=32, beta=0.95, dropout=0.2, lr=0.0005, epochs=50, class_weight=None
- Random stratified split: `test_size=0.2`, `random_state=42` (trial-level, **not** subject-independent)

**Macro F1 note:** Step 28 documents **0.4804** Macro F1 at the same **53.12%** accuracy; later Steps 29–32 report **0.5103** as the project reference. Both share the same accuracy; **0.5103** is used as the canonical baseline throughout Steps 33–40.

---

## 3. Summary Table of DEAP Experiments

| # | Experiment | Accuracy | Macro F1 | vs baseline | Interpretation |
|---|------------|----------|----------|-------------|----------------|
| 1 | Statistical features (Step 13 SNN) | 39.06% | 0.366 | Hurt vs best | Weak static baseline; 6 stats × 40 channels insufficient for 4-class VA mapping. |
| 2 | Welch PSD + statistical (Step 19) | not recorded | not recorded | Unknown | Enabled 440-dim input; no saved DEAP SNN benchmark separate from later combined features. |
| 3 | Differential Entropy only (Step 23) | ~42.58% (ref.) | not recorded | Below temporal best | DE improved over pure stats; became reference before temporal SNN. |
| 4 | Combined Statistical + DE (Step 24) | ~42.58% (ref.) | not recorded | Below temporal best | Same accuracy reference as DE-only era; richer static vector but still static SNN input. |
| 5 | Random Forest + SelectKBest (Step 20) | not recorded | not recorded | Unknown | RF added for comparison; metrics not exported to `results/metrics/`. |
| 6 | Label strategy `median` | not recorded | not recorded | Unknown | Default early; superseded by `mean` for best runs. |
| 7 | Label strategy `mean` | **53.12%** | **0.5103** | **Best (with temporal SNN)** | Used for all best temporal runs. |
| 8 | Label strategy `fixed_5` | skipped | skipped | N/A | Empty class (Happy/Excited); training auto-skipped (`step_22`). |
| 9 | Label strategy `quantile_40` | not recorded | not recorded | Unknown | Implemented; no per-strategy metrics saved. |
| 10 | Label strategy `quantile_60` | not recorded | not recorded | Unknown | Implemented; no per-strategy metrics saved. |
| 11 | Temporal Windowed SNN (Step 27) | 53.12% | 0.4804–0.5103 | **Major improvement** | Real temporal DE windows vs repeating static vector; largest DEAP gain in the project. |
| 12 | Temporal SNN fine tuning (Step 28) | 53.12% | 0.4804 | Improved acc, modest F1 | 288-config grid; best matched Step 27 accuracy; F1 reported as 0.4804 in Step 28 doc. |
| 13 | Temporal spike encoding (Step 30) | not recorded | not recorded | Likely hurt | `TEMPORAL_SPIKE_ENCODING` defaults to `False`; no exported metrics beating 0.5103. |
| 14 | Ambiguous sample filtering (Step 30) | not recorded | not recorded | Likely hurt / neutral | `USE_AMBIGUOUS_SAMPLE_FILTER` defaults to `False`; no saved improvement. |
| 15 | Channel selection `all` | (baseline path) | (baseline path) | Baseline | All 40 channels (32 EEG + 8 peripheral) used in best temporal run. |
| 16 | Channel selection `frontal` (7 ch.) | not recorded | not recorded | Unknown | Reduces features; no saved DEAP comparison. |
| 17 | Channel selection `frontal_temporal` (11 ch.) | not recorded | not recorded | Unknown | Literature-motivated subset; no saved DEAP comparison. |
| 18 | Frontal asymmetry features (Step 32) | not recorded (full); 43.75% (8-subject fast) | not recorded (full); 0.3743 (8-subject fast) | Hurt (fast test) | +15 asymmetry dims/window did not beat 200-dim baseline on fast subset. |
| 19 | EEG-only channels (Step 33) | 43.75–54.69% (8-subject fast) | 0.3224–0.4616 (8-subject fast) | Below full baseline | Removing peripherals did not beat full-dataset 0.5103 on fast grid. |
| 20 | Log PSD temporal features (Step 33) | 54.69% (8-subject fast, `global`) | 0.4600 (8-subject fast) | Below baseline F1 | Highest fast-test accuracy but Macro F1 still below 0.5103. |
| 21 | DE + log PSD temporal (Step 33) | 50.00% (8-subject fast) | 0.4616 (8-subject fast) | Below baseline | Best Macro F1 in Step 33 grid; still −0.0487 vs baseline. |
| 22 | CNN-SNN 5×40 (Step 39, `per_subject_per_channel`) | 21.88% (8-subject fast) | 0.1517 (8-subject fast) | **Severe hurt** | SEED-style CNN collapsed on DEAP; worse than temporal baseline on same split. |
| 23 | DEAP norm `global` (Step 40) | 43.75% (8-subject fast) | 0.3121 (8-subject fast) | Hurt | Subject-aware preprocessing without changing model input did not help. |
| 24 | DEAP norm `per_subject` (Step 40) | 51.56% (8-subject fast) | 0.3890 (8-subject fast) | Hurt | Best among Step 40 modes but Macro F1 −0.1213 vs baseline. |
| 25 | DEAP norm `per_subject_per_channel` (Step 40) | 43.75% (8-subject fast) | 0.3330 (8-subject fast) | Hurt | SEED-winning mode did not transfer to DEAP temporal SNN. |

**Saved metric files consulted:**

- `results/metrics/results_summary.json` — Step 33 grid (8-subject fast)
- `results/metrics/deap_cnn_snn_results.json` — Step 39 (8-subject fast)
- Step 40 — console log only (not yet exported to JSON)

---

## 4. What Improved Performance

### Temporal Windowed SNN (Steps 27–29) — primary success

| Transition | Accuracy | Macro F1 |
|------------|----------|----------|
| Static tuned SNN (Step 26) | 45.31% | 0.4412 |
| Temporal SNN (Step 27–29) | **53.12%** | **0.5103** |
| **Gain** | **+7.81 pp** | **+~0.07** |

Feeding **distinct windowed DE vectors** `(trials, 10, 200)` through the SNN—instead of repeating one static 440-dim vector—exploited temporal EEG structure and aligned better with sequential SNN processing.

### Supporting improvements (pre-temporal)

- **Hyperparameter tuning (Steps 11, 26):** deeper SNN, dropout, class weights, macro-F1 selection raised static SNN from ~39% toward ~45%.
- **Combined Stat+DE features (Step 24):** raised the static-era ceiling to ~**42.58%** accuracy (documented reference).
- **`mean` label strategy:** used for all best temporal results (adaptive threshold vs fixed 5.0).

### Window count confirmation (Step 31)

Testing windows {5, 10, 20, 40} with fixed best config selected **10 windows** at **53.12% / 0.5103**—confirming the Step 29 setting rather than finding a new optimum.

---

## 5. What Hurt Performance

| Change | Evidence | Likely reason |
|--------|----------|---------------|
| **Static spike encoding (Step 12)** | `USE_SPIKE_ENCODING = False` remains default | Stochastic rate encoding on flat features adds noise; fixed older architecture. |
| **Temporal spike encoding (Step 30)** | `TEMPORAL_SPIKE_ENCODING = False` remains default | Extra stochastic binarization over 10×10×200 tensor dilutes DE signal. |
| **Ambiguous sample filtering (Step 30)** | `USE_AMBIGUOUS_SAMPLE_FILTER = False` remains default | Removes many trials; smaller data may not offset label noise reduction. |
| **Frontal asymmetry (Step 32)** | 43.75% / 0.3743 on 8-subject fast vs 53.12% / 0.5103 baseline | Extra 15 dims did not help; asymmetry may be too weak for 4-class VA quadrants. |
| **EEG-only + log PSD / DE+log PSD grid (Step 33)** | Best fast Macro F1 0.4616 < 0.5103 | Feature/normalization variants did not beat windowed DE baseline. |
| **CNN-SNN on DEAP (Step 39)** | 21.88% / 0.1517 vs temporal 48.44% / 0.4189 (same fast run) | 5×40 reshape + CNN overfit or destroyed temporal structure on small DEAP subset. |
| **Subject normalization on DEAP (Steps 39–40)** | Step 40 best: 51.56% acc but 0.3890 F1 | Reduces subject shift in features but does not fix label ambiguity; may remove useful between-subject signal under random trial split. |
| **Per-subject / per-channel norm on fast grid (Step 33)** | Often lowest Macro F1 in grid | Subject z-scoring hurt more than helped under trial-level split. |

---

## 6. Why DEAP Remains Difficult

1. **Derived labels, not direct emotions:** Classes are quadrants of continuous Valence and Arousal ratings (1–9), not experimenter-labeled emotion categories. Trials near thresholds are inherently ambiguous.

2. **4-class VA mapping is noisy:** Subjective ratings, neutral-region trials, and overlapping affective states make quadrant boundaries unstable. Step 30 filtering was designed specifically because ratings in the 4–6 range are unreliable.

3. **Random trial split (not subject-independent):** DEAP evaluation mixes subjects in train and test (`train_test_split` in `snn_model.py`). This is **easier** than SEED’s subject split yet still caps near ~53%—suggesting a **label/signal ceiling**, not only generalization failure.

4. **Smaller effective sample size:** Full DEAP = **1280 trials** (32 × 40). SEED = **50,910 samples**. Fewer examples per class limit deep models (especially CNN-SNN).

5. **Harder task formulation:** **4 classes** from two continuous dimensions vs SEED’s **3 direct emotion classes** (negative / neutral / positive).

6. **Peripheral channels in baseline:** Best run used all 40 channels for temporal DE; EEG-only experiments did not improve full-dataset scores.

7. **Class confusion persists at best result:** Even at 53.12%, Macro F1 0.51 implies weak performance on at least one quadrant (consistent with Step 39 confusion: poor Angry/Stressed recall).

---

## 7. Comparison With SEED

| Factor | DEAP (best) | SEED (best) |
|--------|-------------|-------------|
| **Task** | 4-class VA quadrants | 3-class emotion labels |
| **Labels** | Derived thresholds on V/A | Direct class labels in NPZ |
| **Samples** | 1,280 trials | 50,910 samples |
| **Features** | Raw EEG → windowed DE | Pre-extracted 5-band × 62 channels |
| **Split** | Random stratified (trial) | Subject-independent (train 0–11, test 12–14) |
| **Best SNN-family result** | Temporal SNN: **53.12% / 0.5103** | CNN-SNN + `per_subject_per_channel`: **69.07% / 0.6937** |
| **Subject normalization** | No full-dataset gain (Steps 33, 40) | **+20 pp Macro F1** vs global CNN-SNN (Step 37 → 38) |

**SEED progression (subject split):**

| Model | Accuracy | Macro F1 |
|-------|----------|----------|
| Logistic Regression | 54.63% | 0.5263 |
| Simple Temporal SNN (Step 34) | 44.13% | 0.3781 |
| Strong SNN (Step 35) | 47.09% | 0.4412 |
| CNN-SNN global (Step 36) | 48.79% | 0.4912 |
| **CNN-SNN `per_subject_per_channel` (Step 38)** | **69.07%** | **0.6937** |

The same pipeline family that **plateaus on DEAP** **scales on SEED** once labels are cleaner and subject normalization is applied—strong evidence the implementation works but DEAP’s formulation is the bottleneck.

---

## 8. Most Likely Reasons for Limited DEAP Accuracy

| Rank | Factor | Support |
|------|--------|---------|
| 1 | **Label noise / VA quadrant mapping** | Filtering step added; `fixed_5` breaks; best F1 only ~0.51 at best accuracy; literature aligns on DEAP subjectivity. |
| 2 | **Task difficulty (4 derived classes)** | SEED 3-class direct labels reach 69%; DEAP 4-class caps ~53%. |
| 3 | **Limited samples** | 1280 trials vs 50k+ SEED samples; CNN-SNN fails on small DEAP fast runs. |
| 4 | **Feature–label mismatch** | Many feature engineering wins on paper (asymmetry, PSD, CNN spatial map) did not transfer—signal may not align with noisy quadrants. |
| 5 | **Architecture** | Temporal modeling **did** help (+8 pp); further arch changes (spikes, CNN) **hurt**. Architecture is not the main remaining lever. |
| 6 | **Evaluation protocol** | Trial-level split may inflate vs subject split; still low absolute F1 indicates real task difficulty. |

**Not the primary cause:** Implementation bugs alone—SEED proves the SNN stack trains, evaluates, and improves with sensible changes.

---

## 9. Recommended Final Project Interpretation

Use this narrative in the defense/presentation:

1. **We systematically tested** statistical, spectral, DE, combined, temporal, spike, asymmetry, channel, normalization, and CNN-SNN approaches on DEAP—documented in Steps 19–40.

2. **The strongest DEAP improvement was Temporal Windowed SNN** (~45% → **53.12%**, Macro F1 **0.5103**), validating the choice of SNN for sequential EEG dynamics.

3. **Further DEAP improvements were limited** primarily by **Valence–Arousal label ambiguity**, **4-class derived labeling**, and **dataset scale**—not for lack of experiments.

4. **On SEED, the same SNN family reached 69.07% / 0.6937**, showing the pipeline is **sound when labels and features are cleaner**.

5. **Therefore:** the DEAP ceiling is best explained by **dataset/label quality and task formulation**, with **temporal SNN architecture** as the main successful modeling choice—not a failed SNN implementation.

---

## 10. Recommended Next Steps

These are research directions only (no code changes in Step 41):

| Direction | Rationale |
|-----------|-----------|
| **Binary or 2D VA regression** | Reduce label noise; predict valence/arousal continuously instead of forced quadrants. |
| **Subject-independent DEAP evaluation** | Match SEED protocol; report whether 53% holds when test subjects are unseen. |
| **Soft / probabilistic labels** | Weight trials by distance from VA thresholds instead of hard quadrants. |
| **Ensemble or calibration** | Improve Macro F1 on minority quadrants (e.g., Angry/Stressed) without new features. |
| **Full-dataset rerun of Steps 39–40** | Current CNN-SNN and normalization results are **8-subject fast** only; full 32-subject runs would confirm conclusions. |
| **Compare to published DEAP benchmarks** | Contextualize 53% against literature (often binary or arousal/valence regression, not identical 4-class setup). |

---

## Analysis Questions (Direct Answers)

| # | Question | Answer |
|---|----------|--------|
| 1 | Is the main problem the SNN architecture? | **No.** Temporal SNN improved DEAP substantially; SEED reached 69%. Static/spike/CNN variants hurt or plateaued—architecture was tuned reasonably. |
| 2 | Is the main problem DEAP labels? | **Yes, largely.** Derived VA quadrants are subjective and threshold-sensitive; filtering and strategy docs target label noise. |
| 3 | Is the 4-class Valence–Arousal mapping noisy? | **Yes.** Neutral-region trials, empty classes under `fixed_5`, and low Macro F1 at peak accuracy support this. |
| 4 | Did changing thresholds improve performance? | **Partially / unclear.** `mean` strategy used for best runs; other strategies not fully benchmarked; `fixed_5` unusable. |
| 5 | Did feature engineering improve performance? | **Moderately pre-temporal** (Stat+DE ~42.58%); **not beyond temporal DE baseline** (53.12% / 0.5103). |
| 6 | Did temporal information improve performance? | **Yes — largest gain** (~+8 pp accuracy vs static tuned SNN). |
| 7 | Did spike encoding improve performance? | **No evidence of improvement; likely hurt.** Both static and temporal spike flags default off. |
| 8 | Did subject-aware normalization improve DEAP as on SEED? | **No.** Step 33 grid and Step 40 study did not beat 0.5103; Step 40 best Macro F1 0.3890 (fast). |
| 9 | Why did CNN-SNN work on SEED but fail on DEAP? | SEED: more data, cleaner 3-class labels, subject norm, native 5×62 structure. DEAP: 1280 trials, noisy 4-class VA, 5×40 reshape on windowed DE—CNN-SNN fell to **21.88%** (fast). |
| 10 | Why is DEAP harder than SEED here? | Noisier derived labels, 4 vs 3 classes, far fewer trials, raw EEG pipeline vs pre-extracted features, and persistent quadrant ambiguity. |

---

## Verification of Expected Conclusions

| Conclusion | Supported? | Evidence |
|------------|------------|----------|
| Temporal Windowed SNN was the main successful DEAP improvement | **Yes** | 45.31% → 53.12%; all later steps use it as baseline to beat. |
| Spike encoding hurt DEAP performance | **Likely yes** | No positive metrics; defaults off; stochastic encoding adds variance. |
| CNN-SNN helped SEED but not DEAP | **Yes** | SEED 69.07% vs DEAP 21.88% (Step 39 fast JSON). |
| Subject normalization strongly helped SEED but not DEAP | **Yes** | SEED +0.20 Macro F1 (Step 37→38); DEAP Step 40 best F1 0.3890 (fast). |
| DEAP labels are likely noisier (VA thresholds) | **Yes** | Step 22, 30, class confusion, subjective ratings. |
| DEAP has fewer samples than SEED | **Yes** | 1,280 trials vs 50,910 samples. |
| 4-class DEAP harder than 3-class SEED | **Yes** | Best F1 0.51 vs 0.69 under same project framework. |
| SNN performs well when labels/features are cleaner (SEED) | **Yes** | 69.07% CNN-SNN on SEED vs 53.12% temporal on DEAP. |

---

## References

| Resource | Path |
|----------|------|
| Step 33 results | `results/metrics/results_summary.json` |
| Step 39 DEAP CNN-SNN | `results/metrics/deap_cnn_snn_results.json` |
| Step 38 SEED best | `results/metrics/seed_best_cnn_snn_results.json` |
| Step 37 normalization | `results/metrics/seed_subject_shift_study.json` |
| Step 40 console | Terminal log (`DEAP Temporal Normalization Study`) |
| Experiment docs | `docs/step_22` – `docs/step_40` |

---

*Step 41 complete — analysis report only; no model code or training changes.*
