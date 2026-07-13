# DEAP Research Archive (Steps 39–45)

Research experiments on the DEAP dataset that did not become the final presentation model. **All code is preserved**; this folder keeps the main project path clean.

## Modules

| File | Step | Description |
|------|------|-------------|
| `deap_cnn_snn.py` | 39 | DEAP CNN-SNN vs Temporal SNN baseline |
| `deap_temporal_normalization_study.py` | 40 | Temporal SNN normalization grid |
| `deap_binary_validation.py` | 43 | Binary Valence/Arousal validation |
| `deap_subject_dependent_snn.py` | 44 | Per-subject stratified CV |
| `deap_asymmetry_snn.py` | 45 | Symmetric-difference asymmetry features |

Analysis-only DEAP studies (label distribution / failure analysis) were documented during development and are kept locally outside the public tree. Experiment metrics remain under `results/metrics/`.

## Imports

Backward-compatible shims remain at the project root:

```python
from src.deap_cnn_snn import run_deap_cnn_snn_experiment  # works unchanged
```

Direct archive import:

```python
from src.archive.deap_research import run_deap_binary_validation
```

## Running

Set in `main.py`:

```python
RUN_PRESENTATION_MODE = False
FINAL_MODE = "experimental"
RUN_DEAP_BINARY_VALIDATION = True  # exactly one flag
```

Project overview, results, and future directions are documented in the root `README.md`.
