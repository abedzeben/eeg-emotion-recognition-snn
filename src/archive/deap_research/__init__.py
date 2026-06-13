"""
DEAP research experiments (Steps 39–45).

Analysis-only steps 41–42 live in docs/step_41_deap_failure_analysis.md
and docs/step_42_label_distribution_analysis.md.

Import from the top-level shims (e.g. src.deap_cnn_snn) for backward compatibility,
or from this package directly for archive-aware code.
"""

from src.archive.deap_research.deap_asymmetry_snn import run_deap_asymmetry_snn
from src.archive.deap_research.deap_binary_validation import run_deap_binary_validation
from src.archive.deap_research.deap_cnn_snn import (
    DEAP_TEMPORAL_BASELINE,
    run_deap_cnn_snn_experiment,
    run_final_dataset_comparison,
)
from src.archive.deap_research.deap_subject_dependent_snn import run_deap_subject_dependent_snn
from src.archive.deap_research.deap_temporal_normalization_study import (
    run_deap_temporal_normalization_study,
)

__all__ = [
    "DEAP_TEMPORAL_BASELINE",
    "run_deap_asymmetry_snn",
    "run_deap_binary_validation",
    "run_deap_cnn_snn_experiment",
    "run_deap_subject_dependent_snn",
    "run_deap_temporal_normalization_study",
    "run_final_dataset_comparison",
]
