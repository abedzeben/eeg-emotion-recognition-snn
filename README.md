# EEG Emotion Recognition using Spiking Neural Networks

Final-year Computer Science capstone project on **EEG-based emotion recognition**, comparing classical machine learning with **Spiking Neural Networks (SNNs)** and hybrid CNN–SNN models on the **DEAP** and **SEED** datasets.

**Final presentation model:** SEED CNN-SNN with subject-aware normalization and subject-independent evaluation  
**Final SEED result:** **69.07% accuracy** · **0.6937 Macro F1**

---

## Project Overview

This repository implements a complete research pipeline for classifying emotions from EEG:

1. Load and preprocess public EEG emotion datasets (DEAP and SEED)
2. Extract statistical, frequency, and temporal features where applicable
3. Train and compare **Logistic Regression**, **SNN**, **Temporal SNN**, and **CNN-SNN** models
4. Evaluate with accuracy, Macro F1, confusion matrices, and subject-aware protocols
5. Export metrics and presentation figures

The default entry point runs the **final SEED CNN-SNN presentation pipeline**. Earlier DEAP experiments remain available through archived research modes.

---

## Motivation

Emotion recognition from brain signals supports affective computing, human–computer interaction, and mental-health monitoring. EEG is noninvasive but noisy and highly variable across subjects. This project asks whether **spiking and hybrid neural models** can compete with classical baselines under realistic evaluation, and how **dataset choice, label design, and normalization** affect performance.

---

## Research Journey

| Phase | Focus | Outcome |
|-------|--------|---------|
| DEAP baseline | Logistic Regression + early SNN on Valence–Arousal labels | Established a classical baseline and an SNN training pipeline |
| DEAP Temporal SNN | Windowed Differential Entropy + Temporal SNN | Best DEAP **4-class** result: **53.12% accuracy / 0.5103 Macro F1** |
| DEAP validation & limits | Binary Valence / Arousal, label and architecture probes | Binary tasks outperformed 4-class mapping; further DEAP changes did not beat the Temporal SNN reference |
| SEED exploration | Stronger SNNs and CNN-SNN on cleaner 3-class labels | Subject-independent evaluation became the reporting standard |
| Final model | CNN-SNN + `per_subject_per_channel` normalization | **69.07% accuracy / 0.6937 Macro F1** (presentation result) |

DEAP research code is preserved under `src/archive/deap_research/` (with compatibility shims under `src/`). The public default path is the SEED presentation mode.

---

## Datasets

**Dataset files are not intended to be distributed with this repository.** Obtain them from their official sources and place them locally as described under [Dataset Setup](#dataset-setup). Respect each dataset’s license and citation requirements.

### DEAP

- **Role:** Extensive experimentation and upper-bound study for 4-class Valence–Arousal emotion mapping
- **Content:** EEG (and peripheral) recordings from **32** participants, **40** trials each
- **Task studied here:** Discrete emotion from continuous Valence / Arousal ratings (4-class; also binary Valence and Arousal validation)
- **Local path:** `data/raw/` (`s01.dat` … `s32.dat`)
- **Official site:** [DEAP Dataset](https://www.eecs.qmul.ac.uk/mmv/datasets/deap/)

### SEED

- **Role:** Final presentation dataset
- **Task:** 3-class emotion (**Negative**, **Neutral**, **Positive**)
- **Local path:** `data/seed/` (NPZ feature / label / subject files expected by the loader)
- **Official information:** SEED / BCMI emotion EEG resources (SJTU)

In this project’s SEED setup, features are organized as frequency-band × channel maps (model input **5 × 62**). Evaluation uses a fixed **subject-independent** split: train subjects **0–11**, test subjects **12–14**.

---

## Processing Pipeline

### DEAP (research / archive path)

```
data/raw/s*.dat
    → load + bandpass filtering + normalization
    → feature extraction (including windowed Differential Entropy for Temporal SNN)
    → Valence–Arousal label mapping (4-class or binary)
    → Logistic Regression / SNN / Temporal SNN
    → metrics export
```

### SEED (default presentation path)

```
data/seed/*.npz
    → load features, labels, subject IDs
    → subject-independent train/test split
    → subject-aware normalization (final: per_subject_per_channel)
    → CNN-SNN training with validation early stopping
    → metrics + figures under results/presentation/
```

---

## Models

| Model | Dataset role | Notes |
|-------|--------------|--------|
| **Logistic Regression** | DEAP + SEED baseline | Classical linear baseline |
| **SNN** | DEAP (early) / SEED variants | Fully connected SNN (snnTorch) |
| **Temporal SNN** | DEAP (best 4-class) | Best DEAP config: windowed DE features + fixed Temporal SNN hyperparameters (`hidden=128`, `second_hidden=32`, `beta=0.95`, `dropout=0.2`, `lr=0.0005`, `epochs=50`) |
| **CNN-SNN** | SEED (final) | CNN on band–channel maps + SNN classifier head; final config uses LR `0.001`, dropout `0.3`, beta `0.95`, balanced class weights, up to 100 epochs, `num_steps=10` |

---

## Evaluation Strategy

- **Metrics:** Accuracy, Macro F1 (primary fairness metric), plus confusion matrices and per-class scores in the presentation export
- **DEAP 4-class / binary Temporal SNN:** Stratified trial-level train/test split (`test_size=0.2`, `random_state=42`) as used in the Temporal SNN experiments
- **SEED final:** Subject-independent split (held-out subjects); not a random shuffle of all samples
- **Normalization (SEED final):** `per_subject_per_channel` — each subject’s channels are scaled with that subject’s training statistics to reduce inter-subject shift

---

## Final Results

### SEED — final CNN-SNN (presentation)

| Metric | Value |
|--------|-------|
| Accuracy | **69.07%** |
| Macro F1 | **0.6937** |
| Normalization | `per_subject_per_channel` |
| Split | Subject-independent (train 0–11, test 12–14) |

Saved exports (after a successful presentation run): `results/presentation/` and `results/metrics/seed_best_cnn_snn_results.json`.

### DEAP — best 4-class Temporal SNN (reference)

| Metric | Value |
|--------|-------|
| Accuracy | **53.12%** |
| Macro F1 | **0.5103** |

### DEAP — binary validation (same Temporal SNN family)

Binary Valence and Arousal classification **outperformed** the 4-class reference on accuracy (saved run in `results/metrics/deap_binary_validation.json`: Valence **69.53%**, Arousal **67.58%**). This supports the conclusion that the SNN pipeline can learn DEAP signal while the **4-class Valence–Arousal mapping** is a major performance bottleneck.

### SEED baselines (for context)

| Model | Accuracy | Macro F1 |
|-------|----------|----------|
| Logistic Regression | 54.63% | 0.5263 |
| Previous CNN-SNN (global normalization) | 48.79% | 0.4912 |
| **Best CNN-SNN (final)** | **69.07%** | **0.6937** |

---

## Key Findings

1. **Temporal modeling mattered on DEAP.** Moving from static features to a Temporal SNN with windowed Differential Entropy produced the strongest DEAP 4-class result (~53% / 0.5103).
2. **4-class VA labels limited DEAP.** Binary Valence/Arousal tasks scored clearly above that ceiling with the same Temporal SNN family, indicating label/task difficulty rather than a total failure of the model class.
3. **SEED enabled a clearer final story.** Cleaner 3-class labels and subject-independent evaluation made results easier to interpret for presentation.
4. **Subject-aware normalization was decisive on SEED.** Switching from global normalization to `per_subject_per_channel` raised CNN-SNN Macro F1 from **0.4912** to **0.6937** under the same hybrid architecture.
5. **CNN-SNN was the best presentation model.** The hybrid used spatial CNN features over band–channel maps and an SNN temporal head; it beat Logistic Regression and earlier SEED SNN/CNN-SNN variants on the held-out subjects.

---

## Repository Structure

```
eeg-emotion-recognition-snn/
├── main.py                 # Entry point (presentation mode by default)
├── requirements.txt
├── README.md
├── data/
│   ├── raw/                # DEAP .dat files (local; not for redistribution)
│   └── seed/               # SEED NPZ files (local; not for redistribution)
├── src/
│   ├── load_data.py
│   ├── preprocessing.py
│   ├── features.py
│   ├── labels.py
│   ├── baseline_model.py
│   ├── snn_model.py
│   ├── seed_experiment.py
│   ├── seed_cnn_snn.py
│   ├── seed_subject_shift_study.py
│   ├── presentation_mode.py
│   ├── project_modes.py
│   ├── evaluate.py / visualize.py / results_export.py
│   ├── deap_*.py           # Compatibility shims → archive
│   └── archive/deap_research/   # Archived DEAP research experiments
└── results/
    ├── presentation/       # Final SEED metrics + figures
    ├── metrics/            # Experiment JSON/CSV trail
    └── figures/            # Earlier experiment plots
```

---

## Installation

**Requirements:** Python 3.9+ (tested with a local virtual environment), PyTorch, and snnTorch.

```bash
git clone https://github.com/abedzeben/eeg-emotion-recognition-snn.git
cd eeg-emotion-recognition-snn

python -m venv .venv

# Windows PowerShell
.\.venv\Scripts\Activate.ps1

# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

Core dependencies: `numpy`, `pandas`, `scipy`, `scikit-learn`, `matplotlib`, `seaborn`, `mne`, `torch`, `snntorch`.

---

## Dataset Setup

Place datasets **locally only**. Do not commit or republish licensed EEG files.

### DEAP

1. Request / download the DEAP preprocessed data from the official DEAP site.
2. Copy subject files into:

```
data/raw/s01.dat
data/raw/s02.dat
...
data/raw/s32.dat
```

### SEED

1. Obtain SEED data from the official SEED / BCMI distribution channels.
2. Prepare NPZ files expected by the loader and place them in `data/seed/`:

```
data/seed/DatasetCaricatoNoImage.npz
data/seed/LabelsNoImage.npz
data/seed/SubjectsNoImage.npz
```

If these files are missing, the presentation run exits with a clear file-not-found message.

---

## Usage

### Default: final presentation (SEED CNN-SNN)

With SEED files in `data/seed/` and the defaults in `main.py`:

```python
RUN_PRESENTATION_MODE = True
```

```bash
python main.py
```

This trains/evaluates the final SEED CNN-SNN configuration and writes presentation outputs under `results/presentation/` (metrics, summary text, and figures), and refreshes the canonical SEED metrics JSON under `results/metrics/`.

### Other modes (optional)

Set `RUN_PRESENTATION_MODE = False`, then use `FINAL_MODE`:

| `FINAL_MODE` | Behaviour |
|--------------|-----------|
| `"seed_best"` | SEED CNN-SNN best configuration (without full presentation export helper path) |
| `"deap_baseline"` | Best DEAP Temporal SNN reference run (requires DEAP files in `data/raw/`) |
| `"compare_results"` | Print SEED vs DEAP comparison from saved metrics / documented references |
| `"experiment_archive"` | List archived experiments (no training) |
| `"experimental"` | Run exactly one enabled archive `RUN_*` flag (DEAP research / legacy studies) |

Archived DEAP experiments can also be gated with `RUN_DEAP_EXPERIMENTS` / `RUN_ARCHIVED_RESEARCH` as documented in `main.py`.

---

## Limitations

- DEAP 4-class performance remains modest (~53%); continuous ratings mapped to quadrants are ambiguous and overlapping.
- The strongest DEAP Temporal SNN reference uses a **random stratified trial split**, not a full subject-independent protocol.
- SEED results depend on the prepared NPZ feature representation and the fixed subject split used in this project.
- Training requires a capable CPU/GPU; full presentation runs are slower than archive “fast” experiment flags.
- Results may vary slightly with hardware and library versions; reported figures are those saved for this project’s reference runs.

---

## Future Work

- Subject-independent or leave-one-subject-out protocols on DEAP for fairer comparison with SEED
- Soft / continuous Valence–Arousal targets instead of hard 4-class quadrants
- Cross-dataset transfer (SEED ↔ DEAP) under compatible feature spaces
- Stronger regularization and calibrated evaluation for imbalanced binary DEAP labels
- Packaging a reproducible environment lockfile (pinned dependency versions)

---

## Authors

This project was developed as a final-year Computer Science capstone project.

**Team Members**

* **Abed Zeben**
* **Omar Gharra**

The project focuses on EEG-based emotion recognition using machine learning, Spiking Neural Networks (SNNs), and hybrid CNN-SNN architectures.

---

## Dataset Citations

Please cite the original dataset papers when using this work or publishing results based on these datasets:

**DEAP**  
S. Koelstra *et al.*, “DEAP: A Database for Emotion Analysis Using Physiological Signals,” *IEEE Transactions on Affective Computing*, vol. 3, no. 1, pp. 18–31, 2012.

**SEED**  
W.-L. Zheng and B.-L. Lu, “Investigating Critical Frequency Bands and Channels for EEG-based Emotion Recognition with Deep Neural Networks,” *IEEE Transactions on Autonomous Mental Development*, vol. 7, no. 3, pp. 162–175, 2015.

Also follow the current citation and license terms on the official DEAP and SEED distribution pages.

---

## License note

This repository contains **code and experiment artifacts**. Dataset ownership remains with the original providers. Dataset files must be obtained separately and used under their respective licenses.
