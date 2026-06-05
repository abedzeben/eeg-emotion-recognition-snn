# Demo Script — EEG Emotion Recognition with SNN

Use this script as a spoken guide during project presentation and judge evaluation.  
**Estimated duration:** 8–12 minutes (adjust as needed).

---

## 1. Project Introduction

> “Good [morning/afternoon]. My project is **EEG Emotion Recognition with Spiking Neural Networks**.
>
> The goal is to classify human emotions from brain signals using the **DEAP dataset**, and to compare a classical machine learning baseline against a **Spiking Neural Network** — a biologically inspired model that processes information through spikes, similar to neurons in the brain.
>
> The pipeline is fully automated: from raw EEG loading, through preprocessing and feature extraction, to model training, evaluation, visualization, and exported results.”

**Key message:** End-to-end, reproducible, baseline vs SNN comparison.

---

## 2. Problem Statement

> “Emotion recognition from EEG is important for human–computer interaction, mental health monitoring, and adaptive systems. The challenge is that EEG signals are noisy, high-dimensional, and emotions are subjective.
>
> We address two classification tasks:
>
> 1. **Binary arousal** — is the subject Calm or Excited?
> 2. **Multi-emotion** — four classes from the Valence–Arousal model: Calm/Relaxed, Happy/Excited, Sad/Low, and Angry/Stressed.
>
> A strong baseline alone can hide class imbalance problems — for example, predicting only the majority class. We therefore report **macro F1** in addition to accuracy, and we compare whether an SNN can offer value on the harder multi-class task.”

**Key message:** Real-world problem, two difficulty levels, fair metrics.

---

## 3. Dataset Overview

> “We use the **DEAP dataset** — 32 participants, 40 music-video trials each, with self-reported ratings.
>
> Each trial includes EEG with shape **40 channels × 8064 samples**. Labels include **Valence, Arousal, Dominance, and Liking**.
>
> We load all subject files `s01.dat` through `s32.dat` from `data/raw/`, concatenate trials across subjects, and train on the combined dataset. This gives us a larger, more realistic evaluation setting than a single subject.”

| Item | Value |
|------|-------|
| Dataset | DEAP |
| Subjects | Up to 32 |
| Trials per subject | 40 |
| EEG shape per trial | (40, 8064) |
| Labels used | Valence (col 0), Arousal (col 1) |

**Key message:** Established benchmark, multi-subject loading.

---

## 4. Processing Pipeline

> “The pipeline has six main stages:
>
> 1. **Load** — `load_all_deap_files()` reads all DEAP `.dat` files  
> 2. **Preprocess** — bandpass filter (0.5–50 Hz) and per-channel normalization  
> 3. **Features** — 240-dimensional feature vector per trial  
> 4. **Labels** — binary arousal or 4-class Valence–Arousal quadrants  
> 5. **Models** — tuned Logistic Regression baseline + tuned SNN  
> 6. **Outputs** — metrics, confusion matrices, CSV/JSON export  
>
> Everything runs from a single command: `python main.py`.”

```
data/raw → load → preprocess → features → labels → train → evaluate → visualize → export
```

**Key message:** Clean, modular, one entry point.

---

## 5. Feature Extraction

> “For each trial and each of the 40 EEG channels, we extract **six features**:
>
> - **Mean** and **variance** in the time domain  
> - **Band power** in Theta, Alpha, Beta, and Gamma using Welch’s method  
>
> That gives **40 × 6 = 240 features** per trial — a compact representation that captures both amplitude and frequency content relevant to emotion.”

| Band | Range (Hz) |
|------|------------|
| Theta | 4–8 |
| Alpha | 8–13 |
| Beta | 13–30 |
| Gamma | 30–45 |

**Key message:** Interpretable, EEG-relevant features.

---

## 6. Binary Classification Results

> “For **binary arousal** — Calm vs Excited — we threshold Arousal at 5.
>
> Results on the held-out test split:

| Model | Accuracy | Macro F1 |
|-------|----------|----------|
| Baseline Logistic Regression | **0.750** | **0.602** |
| Tuned SNN | 0.742 | 0.596 |

> “The **baseline slightly wins** on binary classification. Both models perform reasonably well, which shows the feature pipeline is effective for a simpler two-class task. Macro F1 confirms we are not only predicting one class.”

**Talking point:** Binary task validates the pipeline; baseline is strong here.

---

## 7. Multi-Emotion Classification Results

> “The **primary task** is harder: **four emotion classes** from Valence and Arousal.
>
> We use **adaptive median thresholds** instead of a fixed score of 5. Fixed thresholds caused a missing Happy/Excited class; median split gives balanced quadrants and enables proper 4-class training.”

| Class | Emotion | Rule |
|-------|---------|------|
| 0 | Calm / Relaxed | V > median, A ≤ median |
| 1 | Happy / Excited | V > median, A > median |
| 2 | Sad / Low | V ≤ median, A ≤ median |
| 3 | Angry / Stressed | V ≤ median, A > median |

**Results:**

| Model | Accuracy | Macro F1 |
|-------|----------|----------|
| Baseline Logistic Regression | 0.383 | 0.362 |
| Tuned SNN | **0.402** | **0.380** |

> “Accuracy is lower than binary — expected for four overlapping emotion categories. Importantly, the **tuned SNN slightly outperforms the baseline** on both accuracy and macro F1. This is where the SNN contributes most in our project.”

**Talking point:** SNN adds value on the harder task.

---

## 8. Visualization Explanation

> “After each run, the pipeline saves professional figures to `results/figures/`:
>
> - **Confusion matrices** for binary and multi-emotion, baseline and SNN  
> - **Bar charts** comparing Baseline vs SNN accuracy and macro F1  
>
> Confusion matrices show *which* emotions are confused — not just overall accuracy. For example, Sad vs Angry may overlap because both have low Valence but differ in Arousal.”

| File | Purpose |
|------|---------|
| `binary_baseline_cm.png` | Binary baseline errors |
| `binary_snn_cm.png` | Binary SNN errors |
| `multi_baseline_cm.png` | 4-class baseline errors |
| `multi_snn_cm.png` | 4-class SNN errors |
| `accuracy_comparison.png` | Multi-emotion accuracy bars |
| `macrof1_comparison.png` | Multi-emotion macro F1 bars |

**Key message:** Visual evidence supports the numbers.

---

## 9. Final Conclusions

> “In summary:
>
> 1. We built a **complete EEG emotion recognition pipeline** on DEAP.  
> 2. **Binary arousal** classification works well; the Logistic Regression baseline reaches **75% accuracy**.  
> 3. **Multi-emotion** classification is harder but more realistic; our **tuned SNN beats the baseline** (~40% vs ~38% accuracy).  
> 4. We use **fair metrics** (macro F1), **class balancing**, and **hyperparameter tuning**.  
> 5. Results are **exported** to CSV/JSON and **visualized** for reporting.  
>
> **Future work** includes subject-independent evaluation, richer temporal features, and a live Streamlit demo.
>
> Thank you. I’m happy to answer questions or run `python main.py` live.”

**Closing line:** Pipeline works, SNN shows promise on multi-class, project is reproducible and documented.

---

## Quick Reference — One-Liner Answers

| Question | Answer |
|----------|--------|
| Why DEAP? | Standard EEG emotion benchmark with Valence/Arousal labels |
| Why SNN? | Biologically inspired; tested on harder multi-class task |
| Why macro F1? | Fair when classes are imbalanced |
| Why median thresholds? | Fixed threshold 5 caused empty Happy/Excited class |
| How to reproduce? | `pip install -r requirements.txt` then `python main.py` |
