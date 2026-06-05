# Demo Flow — Step-by-Step Presentation Sequence

Follow this order during live demo or judge evaluation.  
**Total time:** ~10 minutes (flexible).

---

## Overview

| Step | Topic | Time |
|------|-------|------|
| 1 | Open GitHub repository | 30 s |
| 2 | Explain project goal | 1 min |
| 3 | Show project structure | 1 min |
| 4 | Show generated figures | 2 min |
| 5 | Show exported results | 1 min |
| 6 | Explain Binary results | 1.5 min |
| 7 | Explain Multi-Emotion results | 2 min |
| 8 | Explain SNN contribution | 1.5 min |
| 9 | Show conclusions | 1 min |

---

## Step 1 — Open GitHub Repository

**Action:** Open the project repository in a browser (or local folder if offline).

**Say:**
> “Here is the full project repository — code, documentation, results, and README are all version-controlled and reproducible.”

**Show:**
- Repository name and description
- `README.md` preview on GitHub
- Recent commit / project activity (if applicable)

---

## Step 2 — Explain Project Goal

**Action:** Scroll to README **Project Overview**.

**Say:**
> “We classify emotions from EEG brain signals using the DEAP dataset. We compare a tuned Logistic Regression baseline against a Spiking Neural Network. The pipeline runs end-to-end from one command.”

**Highlight:**
- Binary arousal (Calm vs Excited)
- 4-class Valence–Arousal emotions (primary task)
- Automated evaluation, plots, and export

---

## Step 3 — Show Project Structure

**Action:** Open file tree in IDE or GitHub.

**Say:**
> “The structure is modular: `src/` for logic, `data/raw/` for DEAP files, `results/` for outputs, `docs/` for step-by-step development notes.”

**Point to:**

```
src/load_data.py      → DEAP loading
src/preprocessing.py  → filter + normalize
src/features.py       → 240-dim features
src/labels.py         → binary + multi-emotion labels
src/baseline_model.py → tuned Logistic Regression
src/snn_model.py      → tuned SNN (+ optional spike encoding)
main.py               → single entry point
```

**Optional:** Briefly open `main.py` — show `USE_SPIKE_ENCODING` flag.

---

## Step 4 — Show Generated Figures

**Action:** Open `results/figures/` folder.

**Order to present:**

1. **`multi_baseline_cm.png`** — “Baseline confusion matrix for 4 emotions”
2. **`multi_snn_cm.png`** — “SNN confusion matrix — compare diagonal vs off-diagonal”
3. **`accuracy_comparison.png`** — “SNN slightly higher accuracy on multi-emotion”
4. **`macrof1_comparison.png`** — “Same trend for macro F1”
5. **`binary_baseline_cm.png`** / **`binary_snn_cm.png`** — “Binary task — both models strong”

**Say:**
> “Confusion matrices show *where* mistakes happen, not just how many. Bar charts summarize Baseline vs SNN at a glance.”

---

## Step 5 — Show Exported Results

**Action:** Open `results/metrics/results_summary.csv` (Excel) or `results_summary.json`.

**Say:**
> “Every experiment is exported automatically — four rows: binary baseline, binary SNN, multi baseline, multi SNN. This supports reports and reproducibility.”

**Verify 4 records:**

| task | model |
|------|-------|
| Binary Classification | Baseline Logistic Regression |
| Binary Classification | Tuned SNN |
| Multi-Emotion Classification | Baseline Logistic Regression |
| Multi-Emotion Classification | Tuned SNN |

---

## Step 6 — Explain Binary Results

**Action:** Reference CSV or README results table.

**Say:**
> “Binary arousal uses Arousal > 5 for Excited. The baseline reaches **75% accuracy** and **0.602 macro F1**. The SNN is close at **74.2%** and **0.596 macro F1**. The baseline wins on this simpler task — that validates our feature pipeline.”

| Model | Accuracy | Macro F1 |
|-------|----------|----------|
| Baseline | 0.750 | 0.602 |
| Tuned SNN | 0.742 | 0.596 |

**Point to:** `binary_*_cm.png` — relatively clean diagonal.

---

## Step 7 — Explain Multi-Emotion Results

**Action:** Show `multi_*_cm.png` and metrics.

**Say:**
> “Four classes from Valence and Arousal quadrants. We use **median thresholds** so all four classes have samples — fixed threshold 5 had zero Happy/Excited trials.
>
> Accuracy is lower (~38–40%) because four emotions overlap in EEG space. The **SNN wins**: **40.2% accuracy** and **0.380 macro F1** vs baseline **38.3%** and **0.362**.”

| Model | Accuracy | Macro F1 |
|-------|----------|----------|
| Baseline | 0.383 | 0.362 |
| Tuned SNN | 0.402 | 0.380 |

**Mention class mapping:** Calm, Happy, Sad, Angry/Stressed (4 quadrants).

---

## Step 8 — Explain SNN Contribution

**Action:** Open `src/snn_model.py` briefly (optional) or use README architecture description.

**Say:**
> “The SNN uses snntorch — Linear layers with Leaky Integrate-and-Fire neurons. We tuned hidden size, learning rate, epochs, and class weights. Grid search selects the best config by macro F1.
>
> **Contribution:** On binary, baseline is slightly better. On **multi-emotion**, the SNN is slightly better — suggesting spiking dynamics help when the task is harder and classes are more balanced.
>
> We also implemented optional **rate spike encoding** (Step 12) for future experiments; default mode uses the tuned static-feature SNN.”

**Do not** run full training live unless time allows — show pre-computed results instead.

---

## Step 9 — Show Conclusions

**Action:** Return to README **Results** and **Future Work** sections.

**Say:**
> “We delivered a complete, documented EEG emotion pipeline. Binary classification works well. Multi-emotion is harder but more realistic, and the SNN shows a small but consistent advantage. Results are visualized and exported. Future work: subject-independent evaluation, richer features, and a live demo app.”

**Closing checklist:**
- [ ] Problem stated
- [ ] Dataset explained
- [ ] Pipeline demonstrated
- [ ] Figures shown
- [ ] Metrics stated
- [ ] SNN contribution clarified
- [ ] Questions invited

---

## Optional Live Run (if time + judges request)

```bash
python main.py
```

**Warn audience:** “Training takes several minutes with full DEAP subjects — I will show pre-generated outputs, or we can start a live run in the background.”

**What to highlight during live run:**
- Multi-emotion label distribution (4 classes)
- `Detected num_classes: 4`
- Final comparison summary
- `Visualization completed` + `Results summary exported`

---

## Related Documents

- **Spoken script:** `docs/demo_script.md`
- **Pre-flight checklist:** `docs/demo_checklist.md`
- **Technical details:** `README.md`, `docs/step_13_multi_emotion_classification.md`
