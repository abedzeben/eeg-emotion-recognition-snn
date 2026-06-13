# Step 42 — Label Strategy Distribution Analysis

Analysis-only step: class distributions for all five Valence–Arousal threshold strategies on the **full DEAP dataset** (32 subjects × 40 trials = **1,280 samples**). No model training.

**Data source:** `data/raw/s01.dat` … `s32.dat` via `load_all_deap_files()`.  
**Label logic:** `src/labels.py` — quadrant mapping unchanged from Steps 22–29.

| Class | Name | Rule |
|-------|------|------|
| 0 | Calm / Relaxed | Valence > t_v, Arousal ≤ t_a |
| 1 | Happy / Excited | Valence > t_v, Arousal > t_a |
| 2 | Sad / Low | Valence ≤ t_v, Arousal ≤ t_a |
| 3 | Angry / Stressed | Valence ≤ t_v, Arousal > t_a |

---

## Dataset summary

| Item | Value |
|------|-------|
| Subjects loaded | 32 |
| Trials per subject | 40 |
| Total trials | **1,280** |
| Valence range | 0.10 – 9.00 |
| Arousal range | 0.09 – 9.00 |
| Mean Valence | 3.47 |
| Mean Arousal | 3.73 |
| Median Valence | 3.08 |
| Median Arousal | 3.32 |

DEAP self-report ratings are **skewed toward the low end** of the 1–9 scale. Fixed midpoint thresholds (e.g. 5.0) therefore cut the data very differently from adaptive (mean/median) thresholds.

---

## Strategy: `median`

**Thresholds:** Valence = **3.0800**, Arousal = **3.3150**

| Class | Name | Sample count | Percentage |
|-------|------|--------------|------------|
| 0 | Calm / Relaxed | 253 | 19.77% |
| 1 | Happy / Excited | 386 | 30.16% |
| 2 | Sad / Low | 387 | 30.23% |
| 3 | Angry / Stressed | 254 | 19.84% |
| **Total** | | **1,280** | **100.00%** |

Empty classes: none.

---

## Strategy: `mean`

**Thresholds:** Valence = **3.4655**, Arousal = **3.7301**

| Class | Name | Sample count | Percentage |
|-------|------|--------------|------------|
| 0 | Calm / Relaxed | 267 | 20.86% |
| 1 | Happy / Excited | 330 | 25.78% |
| 2 | Sad / Low | 418 | 32.66% |
| 3 | Angry / Stressed | 265 | 20.70% |
| **Total** | | **1,280** | **100.00%** |

Empty classes: none.

---

## Strategy: `fixed_5`

**Thresholds:** Valence = **5.0000**, Arousal = **5.0000**

| Class | Name | Sample count | Percentage |
|-------|------|--------------|------------|
| 0 | Calm / Relaxed | 269 | 21.02% |
| 1 | Happy / Excited | **0** | **0.00%** |
| 2 | Sad / Low | 713 | 55.70% |
| 3 | Angry / Stressed | 298 | 23.28% |
| **Total** | | **1,280** | **100.00%** |

**Empty classes:** Class 1 (Happy / Excited) — pipeline skips training for this strategy (`step_22`).

Almost no trials exceed **both** Valence > 5 and Arousal > 5 on this dataset, so the Happy quadrant is empty.

---

## Strategy: `quantile_40`

**Thresholds:** Valence = **2.8500** (40th percentile), Arousal = **2.9700** (40th percentile)

| Class | Name | Sample count | Percentage |
|-------|------|--------------|------------|
| 0 | Calm / Relaxed | 225 | 17.58% |
| 1 | Happy / Excited | 541 | 42.27% |
| 2 | Sad / Low | 289 | 22.58% |
| 3 | Angry / Stressed | 225 | 17.58% |
| **Total** | | **1,280** | **100.00%** |

Empty classes: none.

Lower thresholds push many trials into the high-Valence, high-Arousal (Happy) quadrant.

---

## Strategy: `quantile_60`

**Thresholds:** Valence = **3.9600** (60th percentile), Arousal = **3.9600** (60th percentile)

| Class | Name | Sample count | Percentage |
|-------|------|--------------|------------|
| 0 | Calm / Relaxed | 254 | 19.84% |
| 1 | Happy / Excited | 255 | 19.92% |
| 2 | Sad / Low | 527 | 41.17% |
| 3 | Angry / Stressed | 244 | 19.06% |
| **Total** | | **1,280** | **100.00%** |

Empty classes: none.

Higher thresholds concentrate trials in the low-Valence, low-Arousal (Sad) quadrant.

---

## Balance comparison (valid 4-class strategies)

| Strategy | Min class % | Max class % | Spread (max − min) | Normalized entropy† | Std dev of class % |
|----------|-------------|-------------|--------------------|---------------------|--------------------|
| **median** | 19.77% | 30.23% | **10.47 pp** | 0.9843 | 5.20 |
| **mean** | 20.70% | 32.66% | 11.95 pp | **0.9868** | **4.87** |
| quantile_60 | 19.06% | 41.17% | 22.11 pp | 0.9548 | 9.34 |
| quantile_40 | 17.58% | 42.27% | 24.69 pp | 0.9458 | 10.18 |
| fixed_5 | 0.00% | 55.70% | 55.70 pp | 0.7164 | 19.92 |

† Normalized entropy = Shannon entropy / log(4); **1.0** = perfectly balanced four-way split (25% each).

**Ideal balanced reference:** 320 samples per class (25.00% each).

---

## Visual summary

```
Strategy          Calm    Happy   Sad     Angry
                  (0)     (1)     (2)     (3)
─────────────────────────────────────────────────
median            ████    ██████  ██████  ████     ~20–30% each
mean              ████    █████   ███████ ████     Sad slightly high
fixed_5           ████    (empty) ████████ █████   unusable
quantile_40       ███     ████████ ████   ███      Happy heavy
quantile_60       ████    ████    ███████ ████     Sad heavy
```

---

## Conclusions

### 1. Which strategy produces the most balanced classes?

Among **valid four-class strategies** (all classes non-empty):

| Criterion | Winner | Detail |
|-----------|--------|--------|
| **Smallest max–min spread** | **`median`** | 10.47 percentage points (classes range ~20–30%) |
| **Highest normalized entropy** | **`mean`** | 0.9868 (vs 0.9843 for median) |
| **Lowest std dev of class proportions** | **`mean`** | 4.87 pp |

**`median`** and **`mean`** are both highly balanced and nearly equivalent. Either keeps all four classes within roughly **20–33%** of samples. By contrast:

- **`quantile_40`** overweight Happy / Excited (**42.27%**)
- **`quantile_60`** overweight Sad / Low (**41.17%**)
- **`fixed_5`** is **invalid** (0% Happy) and severely imbalanced (56% Sad)

**Recommendation for balance alone:** prefer **`median`** (tightest spread) or **`mean`** (highest entropy); avoid quantile and fixed strategies on this DEAP corpus.

---

### 2. Which strategy was used for the best DEAP result?

The project’s best DEAP Temporal SNN result (**53.12% accuracy / 0.5103 Macro F1**, Steps 27–29, `step_41`) used:

```python
MULTI_LABEL_STRATEGY = "mean"
```

So the best-performing configuration used **`mean`**, not the slightly more spread-optimal **`median`**. Step 22 did not record separate SNN benchmarks per strategy; **`mean`** became the default for temporal experiments from Step 27 onward.

---

### 3. Does class imbalance explain the DEAP performance ceiling?

**Partially, but it is not the main explanation.**

**Evidence against imbalance as the primary ceiling:**

| Observation | Implication |
|-------------|-------------|
| **`mean` distribution is moderately balanced** | Largest class (Sad / Low) is 32.66%; smallest is 20.70% — far from catastrophic skew |
| **Random baseline = 25%** | Best SNN **53.12%** is well above chance; Macro F1 **0.5103** shows non-trivial per-class learning |
| **Best strategy is already near-optimal for balance** | Switching to **`median`** would shift at most ~2–3 pp per class — unlikely to add ~15+ pp accuracy |
| **Severely imbalanced strategies were not tested as best** | `quantile_*` and `fixed_5` were not used for the 53% result |

**Evidence that imbalance still matters somewhat:**

| Observation | Implication |
|-------------|-------------|
| **Sad / Low is consistently the largest class** under `mean` (418 / 1280 trials) | Models may bias toward predicting Sad; hurts Macro F1 for minority quadrants |
| **`fixed_5` shows label–threshold mismatch** | When thresholds don’t match rating distribution, classes collapse or dominate — a form of **label construction** failure, not just training imbalance |
| **Macro F1 < accuracy at peak** (0.51 vs 0.53) | Suggests uneven per-class recall, consistent with mild imbalance plus quadrant ambiguity |

**Overall interpretation (aligned with Step 41):**

Class imbalance under **`mean`** is **mild** (~21–33% per class). It may contribute to Macro F1 staying near **0.51**, but it **does not fully explain** why accuracy stalls near **53%**. The stronger limits are:

1. **Label noise** — continuous V/A ratings forced into discrete quadrants  
2. **Neutral-region ambiguity** — trials near thresholds don’t map cleanly to one emotion  
3. **Task difficulty** — 4 derived classes vs cleaner direct labels (as on SEED)

Improving balance alone (e.g. switching `mean` → `median`) is unlikely to break the ceiling without addressing **label quality**, not just class counts.

---

## How to reproduce

```bash
python -c "
from pathlib import Path
from src.load_data import load_all_deap_files
from src.labels import compare_label_strategies
_, y = load_all_deap_files(Path('data/raw'))
compare_label_strategies(y)
"
```

Or use `compare_label_strategies(y)` from the main pipeline when DEAP data is loaded.

---

## References

| Resource | Role |
|----------|------|
| `src/labels.py` | Threshold strategies and quadrant mapping |
| `docs/step_22_label_strategy_comparison.md` | Strategy definitions |
| `docs/step_41_deap_failure_analysis.md` | Best result used `mean`; label noise discussion |
| `main.py` | `MULTI_LABEL_STRATEGY = \"mean\"` |

---

*Step 42 complete — distribution analysis only; no model code or training changes.*
