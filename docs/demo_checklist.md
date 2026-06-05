# Demo Checklist — Before Presentation

Use this checklist **the day before** and **1 hour before** your presentation.

---

## Before Presentation

### Hardware & Environment

- [ ] Laptop fully charged (or charger packed)
- [ ] Power adapter and extension cord if needed
- [ ] USB backup drive prepared (see Backup section)
- [ ] Internet access tested (if showing GitHub live)
- [ ] Projector / screen sharing tested

### Project Opens Successfully

- [ ] Repository opens in IDE (Cursor / VS Code)
- [ ] Virtual environment activates without errors
- [ ] `pip install -r requirements.txt` completed (or already installed)
- [ ] No missing imports (`numpy`, `torch`, `snntorch`, `seaborn`, etc.)

### Pipeline Runs

- [ ] `python main.py` runs end-to-end without errors
- [ ] DEAP files present in `data/raw/` (`s*.dat`)
- [ ] Console shows: preprocessing → features → training → evaluation
- [ ] Console ends with:
  - [ ] `Visualization completed`
  - [ ] `Results summary exported`

### Figures Available

Confirm all files exist in `results/figures/`:

- [ ] `binary_baseline_cm.png`
- [ ] `binary_snn_cm.png`
- [ ] `multi_baseline_cm.png`
- [ ] `multi_snn_cm.png`
- [ ] `accuracy_comparison.png`
- [ ] `macrof1_comparison.png`

### Results Exported

Confirm files in `results/metrics/`:

- [ ] `results_summary.csv` — opens in Excel / spreadsheet viewer
- [ ] `results_summary.json` — valid JSON, 4 experiment records

### Documentation Ready

- [ ] `README.md` updated with latest results
- [ ] `docs/demo_script.md` reviewed (this presentation guide)
- [ ] `docs/demo_flow.md` reviewed (step sequence)
- [ ] Key step docs available if judges ask (Steps 13, 14, 15)

### Backup Copy on USB

Copy to USB drive:

- [ ] Full project folder (or zip archive)
- [ ] `results/figures/` (all PNG files)
- [ ] `results/metrics/` (CSV + JSON)
- [ ] `README.md`
- [ ] `docs/demo_script.md`
- [ ] PDF export of README or slides (optional)

---

## 1 Hour Before — Final Check

- [ ] Run `python main.py` once more (or confirm last run succeeded)
- [ ] Open 2–3 key figures to verify they display correctly
- [ ] Open `results_summary.csv` — verify 4 rows, correct metrics
- [ ] Close unnecessary apps to free RAM (SNN training is memory-intensive)
- [ ] Terminal font size increased for live demo visibility
- [ ] GitHub repository page loads (if presenting from browser)

---

## Metrics to Know by Heart

| Task | Model | Accuracy | Macro F1 |
|------|-------|----------|----------|
| Binary | Baseline | 0.750 | 0.602 |
| Binary | Tuned SNN | 0.742 | 0.596 |
| Multi-Emotion | Baseline | 0.383 | 0.362 |
| Multi-Emotion | Tuned SNN | 0.402 | 0.380 |

---

## If Something Fails Live

| Problem | Quick fix |
|---------|-----------|
| `No DEAP .dat files found` | Confirm `data/raw/s01.dat` exists |
| `ModuleNotFoundError` | Run `pip install -r requirements.txt` |
| Training too slow | Show pre-generated figures + CSV instead |
| GitHub offline | Use local README + USB backup |
| Wrong metrics | Open `results/metrics/results_summary.json` as source of truth |

---

## Optional Extras

- [ ] Printed confusion matrix figures (A4)
- [ ] Slides with pipeline diagram
- [ ] QR code linking to GitHub repository
- [ ] Short 2-minute backup video of `main.py` run

---

**Presenter:** _______________________  
**Date:** _______________________  
**All items checked:** [ ] Yes
