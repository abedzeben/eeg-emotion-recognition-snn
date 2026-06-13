from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

from src.deap_cnn_snn import DEAP_TEMPORAL_BASELINE
from src.seed_subject_shift_study import run_seed_best_cnn_snn

FINAL_MODES = (
    "seed_best",
    "deap_baseline",
    "compare_results",
    "experiment_archive",
    "experimental",
)

SEED_BEST_RESULTS_PATH = Path("results/metrics/seed_best_cnn_snn_results.json")
DEAP_BASELINE_RESULTS_PATH = Path("results/metrics/deap_temporal_baseline_results.json")

EXPERIMENT_ARCHIVE_CATALOG: tuple[dict[str, str], ...] = (
    {
        "id": "legacy_deap_pipeline",
        "flag": "RUN_LEGACY_DEAP_PIPELINE",
        "step": "Steps 1–31",
        "description": "Full DEAP multi-emotion pipeline (LR/RF/SNN, figures, export)",
    },
    {
        "id": "seed_experiment",
        "flag": "RUN_SEED_EXPERIMENT",
        "step": "Steps 34–36",
        "description": "SEED SNN / strong SNN / CNN-SNN grid experiments",
    },
    {
        "id": "seed_subject_shift",
        "flag": "RUN_SEED_SUBJECT_SHIFT_STUDY",
        "step": "Step 37",
        "description": "SEED CNN-SNN normalization study (6 modes)",
    },
    {
        "id": "deap_cnn_snn",
        "flag": "RUN_DEAP_CNN_SNN",
        "step": "Step 39",
        "description": "DEAP CNN-SNN vs Temporal SNN baseline",
    },
    {
        "id": "deap_temporal_norm",
        "flag": "RUN_DEAP_TEMPORAL_NORMALIZATION_STUDY",
        "step": "Step 40",
        "description": "DEAP Temporal SNN normalization study (no CNN)",
    },
    {
        "id": "deap_binary_validation",
        "flag": "RUN_DEAP_BINARY_VALIDATION",
        "step": "Step 43",
        "description": "Binary Valence/Arousal Temporal SNN validation",
    },
    {
        "id": "deap_subject_dependent",
        "flag": "RUN_DEAP_SUBJECT_DEPENDENT_SNN",
        "step": "Step 44",
        "description": "Subject-dependent 4-class Temporal SNN (per-subject CV)",
    },
    {
        "id": "deap_asymmetry",
        "flag": "RUN_DEAP_ASYMMETRY_SNN",
        "step": "Step 45",
        "description": "Symmetric difference asymmetry Temporal SNN",
    },
    {
        "id": "snn_research",
        "flag": "RUN_SNN_RESEARCH_EXPERIMENTS",
        "step": "Step 33",
        "description": "DEAP EEG-only temporal SNN preprocessing grid (9 runs)",
    },
    {
        "id": "temporal_window_opt",
        "flag": "RUN_TEMPORAL_WINDOW_OPTIMIZATION",
        "step": "Step 31",
        "description": "Compare temporal window counts for DEAP SNN",
    },
)


def print_experiment_archive() -> None:
    """List archived research experiments (does not run training)."""
    print("\n" + "=" * 60)
    print("Experiment Archive (research / failed paths — not final modes)")
    print("=" * 60)
    print(
        "To run one archived experiment:\n"
        "  1. Set FINAL_MODE = \"experimental\"\n"
        "  2. Enable exactly one RUN_* flag listed below\n"
        "  3. Run: python main.py\n"
    )
    print(f"{'ID':<24} {'Flag':<42} {'Step'}")
    print("-" * 80)
    for entry in EXPERIMENT_ARCHIVE_CATALOG:
        print(f"{entry['id']:<24} {entry['flag']:<42} {entry['step']}")
        print(f"  {entry['description']}")
        print()


def run_final_seed_best(
    *,
    data_dir: str = "data/seed",
    split_mode: str = "subject",
    cnn_snn_num_steps: int = 10,
) -> Dict[str, Any]:
    """Final mode: SEED CNN-SNN + per_subject_per_channel + subject split."""
    print("\n" + "=" * 60)
    print('FINAL_MODE = "seed_best"')
    print("=" * 60)
    print("Model: CNN-SNN")
    print("Normalization: per_subject_per_channel")
    print("Split: subject-independent (train 0–11, test 12–14)")
    return run_seed_best_cnn_snn(
        data_dir=data_dir,
        split_mode=split_mode,
        cnn_snn_num_steps=cnn_snn_num_steps,
        normalization="per_subject_per_channel",
    )


def run_final_deap_baseline(
    folder: str = "data/raw",
    *,
    label_strategy: str = "mean",
    trials_per_subject: int = 40,
) -> Dict[str, Any]:
    """Final mode: best DEAP Temporal SNN (Step 29 config, full dataset)."""
    from src.load_data import load_all_deap_files
    from src.preprocessing import bandpass_filter
    from src.features import TEMPORAL_NUM_WINDOWS, extract_temporal_window_de_features
    from src.labels import EMOTION_LABELS, create_multi_emotion_labels, print_class_distribution
    from src.snn_model import BEST_TEMPORAL_SNN_CONFIG, train_tuned_snn_model
    from src.evaluate import evaluate_classification

    print("\n" + "=" * 60)
    print('FINAL_MODE = "deap_baseline"')
    print("=" * 60)
    print("Model: Temporal SNN (Step 29 best config)")
    print("Features: 10 windows × 200 DE (40 channels × 5 bands)")
    print("Labels:", label_strategy)
    print("Dataset: full DEAP (32 subjects, FAST_TEST_MODE ignored)")
    print("Reference:", DEAP_TEMPORAL_BASELINE)

    X, y_ratings = load_all_deap_files(folder, max_subjects=None)
    X_filtered = bandpass_filter(X)
    n_subjects = X_filtered.shape[0] // trials_per_subject

    y_multi = create_multi_emotion_labels(y_ratings, strategy=label_strategy, verbose=False)

    print("\n=== Dataset Summary ===")
    print("Number of subjects:", n_subjects)
    print("Number of trials:", X_filtered.shape[0])
    print("Class distribution:")
    print_class_distribution(y_multi, EMOTION_LABELS, num_classes=4)

    X_temporal = extract_temporal_window_de_features(
        X_filtered,
        num_windows=TEMPORAL_NUM_WINDOWS,
    )
    print("Temporal feature shape:", X_temporal.shape)
    print("Model config:", BEST_TEMPORAL_SNN_CONFIG)

    (
        _model,
        _X_test,
        y_test,
        y_pred,
        accuracy,
        macro_f1,
        params,
    ) = train_tuned_snn_model(
        X_temporal,
        y_multi,
        temporal=True,
        use_best_temporal_config=True,
        temporal_spike_encoding=False,
        quiet=False,
    )

    evaluate_classification(y_test, y_pred, "DEAP Temporal SNN (final baseline run)")

    print("\n=== vs Step 29 reference ===")
    print(f"Reference accuracy:  {DEAP_TEMPORAL_BASELINE['accuracy']:.4f}")
    print(f"Reference Macro F1: {DEAP_TEMPORAL_BASELINE['macro_f1']:.4f}")
    print(f"This run accuracy:   {accuracy:.4f}  (delta {accuracy - DEAP_TEMPORAL_BASELINE['accuracy']:+.4f})")
    print(
        f"This run Macro F1:   {macro_f1:.4f}  "
        f"(delta {macro_f1 - DEAP_TEMPORAL_BASELINE['macro_f1']:+.4f})"
    )

    payload: Dict[str, Any] = {
        "study": "DEAP Temporal SNN final baseline",
        "final_mode": "deap_baseline",
        "dataset": {
            "subjects": n_subjects,
            "trials": int(X_filtered.shape[0]),
        },
        "label_strategy": label_strategy,
        "reference": DEAP_TEMPORAL_BASELINE,
        "model_config": BEST_TEMPORAL_SNN_CONFIG,
        "feature_shape": list(X_temporal.shape),
        "accuracy": float(accuracy),
        "macro_f1": float(macro_f1),
        "params": params,
    }

    DEAP_BASELINE_RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(DEAP_BASELINE_RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"\nResults saved: {DEAP_BASELINE_RESULTS_PATH}")

    return payload


def run_final_compare_results() -> None:
    """Final mode: load saved SEED + DEAP metrics and print comparison."""
    print("\n" + "=" * 60)
    print('FINAL_MODE = "compare_results"')
    print("=" * 60)

    if not SEED_BEST_RESULTS_PATH.exists():
        raise FileNotFoundError(
            f"SEED results not found: {SEED_BEST_RESULTS_PATH}. "
            'Run FINAL_MODE = "seed_best" first.'
        )

    with open(SEED_BEST_RESULTS_PATH, encoding="utf-8") as f:
        seed_data = json.load(f)
    seed_metrics = seed_data.get("metrics", seed_data)
    seed_acc = float(seed_metrics.get("accuracy", 0.0))
    seed_f1 = float(seed_metrics.get("macro_f1", 0.0))
    seed_norm = seed_data.get("normalization", seed_metrics.get("normalization", "per_subject_per_channel"))

    deap_acc = DEAP_TEMPORAL_BASELINE["accuracy"]
    deap_f1 = DEAP_TEMPORAL_BASELINE["macro_f1"]
    deap_source = "reference (Step 29 documented baseline)"

    if DEAP_BASELINE_RESULTS_PATH.exists():
        with open(DEAP_BASELINE_RESULTS_PATH, encoding="utf-8") as f:
            deap_run = json.load(f)
        deap_acc = float(deap_run.get("accuracy", deap_acc))
        deap_f1 = float(deap_run.get("macro_f1", deap_f1))
        deap_source = str(DEAP_BASELINE_RESULTS_PATH)

    print("\n=== Final Project Comparison ===")
    print(f"{'Dataset / Task':<36} {'Accuracy':>10} {'Macro F1':>10} {'Source'}")
    print("-" * 90)
    print(
        f"{'SEED CNN-SNN (subject split)':<36} {seed_acc:>10.4f} {seed_f1:>10.4f} "
        f"{SEED_BEST_RESULTS_PATH.name}"
    )
    print(
        f"{'DEAP Temporal SNN (4-class)':<36} {deap_acc:>10.4f} {deap_f1:>10.4f} "
        f"{deap_source}"
    )
    print(f"\nSEED normalization: {seed_norm}")
    print(f"DEAP labels: mean Valence–Arousal quadrants (4-class)")
    print("\nInterpretation:")
    print("  SEED: cleaner 3-class labels + subject split + per_subject_per_channel norm")
    print("  DEAP: pooled trial split; ceiling ~53% driven mainly by label ambiguity")
    print(f"\nAccuracy gap (SEED − DEAP): {seed_acc - deap_acc:+.4f}")
    print(f"Macro F1 gap (SEED − DEAP):  {seed_f1 - deap_f1:+.4f}")


def count_enabled_experimental_flags(flags: Dict[str, bool]) -> int:
    return sum(1 for v in flags.values() if v)


def validate_experimental_flags(flags: Dict[str, bool]) -> Optional[str]:
    """Return enabled flag name, or None if zero/multiple enabled."""
    enabled = [name for name, on in flags.items() if on]
    if len(enabled) == 0:
        return None
    if len(enabled) > 1:
        raise ValueError(
            "Enable exactly one experimental archive flag. Enabled: "
            + ", ".join(enabled)
        )
    return enabled[0]
