from src.load_data import load_all_deap_files
from src.preprocessing import (
    NORMALIZATION_MODES,
    bandpass_filter,
    normalize_with_mode,
    print_normalization_mode,
)
from src.features import (
    extract_features,
    extract_temporal_window_de_features,
    extract_temporal_window_snn_features,
    extract_temporal_features_by_type,
    print_feature_mode_comparison,
    print_temporal_snn_feature_info,
    print_temporal_feature_type_info,
    print_frontal_asymmetry_feature_info,
    remove_constant_features,
    FEATURE_MODES,
    get_feature_mode_name,
    get_expected_feature_size,
    TEMPORAL_NUM_WINDOWS,
    TEMPORAL_FEATURE_TYPES,
)
from src.channel_selection import (
    print_channel_selection_info,
    print_eeg_only_channel_info,
    select_channels,
    select_eeg_only_channels,
)
from src.baseline_model import train_baseline_model
from src.random_forest_model import train_random_forest_model
from src.snn_model import (
    train_tuned_snn_model,
    train_spike_encoded_snn_model,
    print_temporal_spike_encoding_info,
)
from src.evaluate import (
    evaluate_classification,
    evaluate_snn_research_experiment,
    print_snn_research_summary,
)
from src.labels import (
    BINARY_LABELS,
    compare_label_strategies,
    create_multi_emotion_labels,
    create_clear_multi_emotion_labels,
    get_empty_classes,
    print_ambiguous_sample_filter_summary,
    print_class_distribution,
    subset_arrays_by_mask,
    EMOTION_LABELS,
)
from src.visualize import generate_all_figures
from src.results_export import export_results_summary
from src.seed_experiment import run_seed_experiment
from src.seed_subject_shift_study import run_seed_subject_shift_study
import os
import numpy as np
from typing import Any, Dict, List, Optional, Tuple

# SNN mode: False = Step 11 tuned SNN (default), True = Step 12 spike-encoded SNN (experimental)
USE_SPIKE_ENCODING = False

# Step 26: reduced SNN hyperparameter grid for fast experiments (64 vs 576 configs)
SNN_FAST_GRID = True

# Step 27: windowed DE features per time step for SNN only (classical models unchanged)
USE_TEMPORAL_SNN_FEATURES = True

# Step 28: focused hyperparameter grid for temporal SNN fine-tuning
TEMPORAL_SNN_FINE_TUNE = False

# Step 29: train single best temporal SNN config (skip grid search)
USE_BEST_TEMPORAL_SNN_CONFIG = True

# Step 29: skip Logistic Regression and Random Forest (SNN only)
RUN_CLASSICAL_MODELS = False

# Set True to also run the legacy binary arousal pipeline (Calm vs Excited)
RUN_BINARY_CLASSIFICATION = False

# Step 19: add Welch PSD band-power features on top of extended statistical features
USE_FREQUENCY_FEATURES = False

# Step 23: Differential Entropy features per frequency band (replaces active feature set when True)
USE_DIFFERENTIAL_ENTROPY = False

# Step 24: combine statistical (240) + Differential Entropy (200) features
USE_COMBINED_STAT_DE_FEATURES = True

# Step 25: subset EEG channels before feature extraction
USE_CHANNEL_SELECTION = True
CHANNEL_SELECTION_MODE = "all"  # "all" | "frontal" | "frontal_temporal"

# Step 21: remove constant / near-constant features before training
REMOVE_CONSTANT_FEATURES = True

# Step 22: multi-emotion Valence-Arousal threshold strategy
MULTI_LABEL_STRATEGY = "mean"

# Step 30: remove ambiguous Valence-Arousal samples near neutral boundary
USE_AMBIGUOUS_SAMPLE_FILTER = False
LOW_THRESHOLD = 4.5
HIGH_THRESHOLD = 5.5

# Step 30 (research): NeuCube-inspired temporal rate spike encoding for SNN
TEMPORAL_SPIKE_ENCODING = False
ENCODING_STEPS = 10

# Step 31: compare temporal window counts for Temporal SNN (best config only)
RUN_TEMPORAL_WINDOW_OPTIMIZATION = False
TEMPORAL_WINDOW_OPTIONS = [5, 10, 20, 40]

# Step 32: frontal EEG asymmetry features for temporal SNN only
USE_FRONTAL_ASYMMETRY_FEATURES = False

# Step 33: EEG-only temporal SNN (exclude peripheral DEAP channels)
SNN_USE_EEG_ONLY_CHANNELS = True

# Step 33: subject-aware normalization (global | per_subject | per_subject_per_channel)
NORMALIZATION_MODE = "global"

# Step 33: temporal SNN feature type (de | log_psd | de_log_psd)
TEMPORAL_FEATURE_TYPE = "de"

# Step 33: run 9 SNN preprocessing experiments (3 norms × 3 feature types)
RUN_SNN_RESEARCH_EXPERIMENTS = False

# Fast experiment mode: load fewer subjects for quick label-strategy tests
FAST_TEST_MODE = True
MAX_SUBJECTS = 8

TRIALS_PER_SUBJECT = 40
SNN_RESEARCH_BASELINE_ACCURACY = 0.5312
SNN_RESEARCH_BASELINE_MACRO_F1 = 0.5103

# Step 34: SEED dataset SNN experiment (additional mode; DEAP pipeline unchanged)
RUN_SEED_EXPERIMENT = True
RUN_SEED_ONLY = True
SEED_DATA_DIR = "data/seed"
SEED_SPLIT_MODE = "subject"  # "trial" | "subject"
SEED_NORMALIZATION_MODE = "train_only_standard"  # "global" | "train_only_standard"

# Step 35/36: SEED SNN mode ("simple" | "strong" | "cnn_snn")
SEED_SNN_MODE = "cnn_snn"
SEED_SNN_FAST_GRID = True
SEED_CNN_SNN_FAST_GRID = True
CNN_SNN_NUM_STEPS = 10

# Step 37: subject shift normalization study (fixed CNN-SNN config, no grid search)
RUN_SEED_SUBJECT_SHIFT_STUDY = True
SEED_SUBJECT_SHIFT_FAST = True


def _build_subject_ids(n_trials: int, trials_per_subject: int = TRIALS_PER_SUBJECT) -> np.ndarray:
    """Map each trial to its DEAP subject index (40 trials per subject file)."""
    if n_trials % trials_per_subject != 0:
        raise ValueError(
            f"Expected trials ({n_trials}) to be a multiple of {trials_per_subject}"
        )
    n_subjects = n_trials // trials_per_subject
    return np.repeat(np.arange(n_subjects), trials_per_subject)


def _prepare_temporal_snn_input(
    X_eeg: np.ndarray,
    *,
    use_eeg_only: bool,
    feature_type: str,
    use_frontal_asymmetry: bool,
    num_windows: int = TEMPORAL_NUM_WINDOWS,
) -> np.ndarray:
    """Bandpass-filtered + normalized EEG → temporal SNN feature tensor."""
    if use_eeg_only:
        X_input, _ = select_eeg_only_channels(X_eeg)
        print_eeg_only_channel_info(X_input.shape[1])
    else:
        X_input = X_eeg

    X_temporal = extract_temporal_features_by_type(
        X_input,
        feature_type=feature_type,
        num_windows=num_windows,
        use_frontal_asymmetry=use_frontal_asymmetry,
    )
    print_temporal_feature_type_info(feature_type, X_temporal, num_windows=num_windows)
    return X_temporal


def _run_snn_research_experiments(
    X_filtered: np.ndarray,
    y_multi: np.ndarray,
    subject_ids: np.ndarray,
) -> List[Dict[str, Any]]:
    """
    Step 33: compare EEG-only temporal SNN preprocessing strategies.

    Runs 3 normalization modes × 3 temporal feature types with fixed best SNN config.
    """
    print("\n=== SNN Research Experiments (Step 33) ===")
    print("SNN_USE_EEG_ONLY_CHANNELS:", SNN_USE_EEG_ONLY_CHANNELS)
    print("USE_BEST_TEMPORAL_SNN_CONFIG: True (fixed)")
    print("TEMPORAL_SPIKE_ENCODING: False")
    print("RUN_CLASSICAL_MODELS: False")
    print("Frontal asymmetry: disabled for research comparison")
    print("Normalization modes:", list(NORMALIZATION_MODES))
    print("Feature types:", list(TEMPORAL_FEATURE_TYPES))
    print(
        "Baseline to beat — Accuracy:",
        f"{SNN_RESEARCH_BASELINE_ACCURACY:.4f},",
        "Macro F1:",
        f"{SNN_RESEARCH_BASELINE_MACRO_F1:.4f}",
    )

    results: List[Dict[str, Any]] = []

    for norm_mode in NORMALIZATION_MODES:
        for feature_type in TEMPORAL_FEATURE_TYPES:
            print(f"\n{'=' * 60}")
            print_normalization_mode(norm_mode)
            print("Feature type:", feature_type)

            X_norm = normalize_with_mode(
                X_filtered,
                norm_mode,
                subject_ids=subject_ids,
            )
            X_temporal = _prepare_temporal_snn_input(
                X_norm,
                use_eeg_only=SNN_USE_EEG_ONLY_CHANNELS,
                feature_type=feature_type,
                use_frontal_asymmetry=False,
            )
            print("Temporal feature shape:", X_temporal.shape)

            _, _, snn_y_test, snn_y_pred, snn_acc, snn_macro_f1, snn_params = (
                train_tuned_snn_model(
                    X_temporal,
                    y_multi,
                    temporal=True,
                    use_best_temporal_config=True,
                    temporal_spike_encoding=False,
                    quiet=True,
                )
            )

            entry = evaluate_snn_research_experiment(
                snn_y_test,
                snn_y_pred,
                normalization_mode=norm_mode,
                feature_type=feature_type,
                feature_shape=X_temporal.shape,
                num_classes=4,
            )
            entry["params"] = snn_params
            results.append(entry)

    print_snn_research_summary(results)

    best = max(results, key=lambda r: r["macro_f1"])
    delta_acc = best["accuracy"] - SNN_RESEARCH_BASELINE_ACCURACY
    delta_f1 = best["macro_f1"] - SNN_RESEARCH_BASELINE_MACRO_F1
    print("\nComparison vs Step 29 baseline (~53.12% / 0.5103):")
    print(f"  Best accuracy delta: {delta_acc:+.4f}")
    print(f"  Best macro F1 delta: {delta_f1:+.4f}")
    if best["macro_f1"] > SNN_RESEARCH_BASELINE_MACRO_F1:
        print("  Result: IMPROVED over baseline Macro F1")
    else:
        print("  Result: did not exceed baseline Macro F1")

    return results


def _run_classification_pipeline(
    X_features,
    y,
    *,
    X_snn_features=None,
    task_name: str,
    num_classes: int,
    run_classical_models: bool = True,
):
    """Train baseline + SNN and evaluate for the given label vector."""
    acc = baseline_macro_f1 = 0.0
    baseline_params: dict = {}
    y_test = y_pred = None

    if run_classical_models:
        model, X_test, y_test, y_pred, acc, baseline_macro_f1, baseline_params = train_baseline_model(
            X_features, y
        )
        print(f"{task_name} baseline model trained")
        evaluate_classification(y_test, y_pred, f"{task_name} Baseline", num_classes=num_classes)

    snn_input = X_snn_features if X_snn_features is not None else X_features
    use_temporal_snn = X_snn_features is not None
    use_best_config = use_temporal_snn and USE_BEST_TEMPORAL_SNN_CONFIG

    if USE_SPIKE_ENCODING:
        print("Running Spike-Encoded SNN (Step 12)")
        snn_model, snn_X_test, snn_y_test, snn_y_pred, snn_acc, snn_macro_f1, snn_params = (
            train_spike_encoded_snn_model(snn_input, y)
        )
        snn_label = f"{task_name} Spike-encoded SNN"
    else:
        if use_best_config:
            if TEMPORAL_SPIKE_ENCODING:
                print("Running Temporal Spike-Encoded SNN (Step 30)")
            else:
                print("Running Temporal Windowed SNN (Step 29 Best Config)")
        elif use_temporal_snn and TEMPORAL_SNN_FINE_TUNE:
            print("Running Temporal Windowed SNN (Step 28 Fine-Tune)")
        elif use_temporal_snn:
            print("Running Temporal Windowed SNN (Step 27)")
        else:
            print("Running Tuned SNN (Step 11)")
        snn_model, snn_X_test, snn_y_test, snn_y_pred, snn_acc, snn_macro_f1, snn_params = (
            train_tuned_snn_model(
                snn_input,
                y,
                snn_fast_grid=SNN_FAST_GRID,
                temporal=use_temporal_snn,
                temporal_fine_tune=use_temporal_snn and TEMPORAL_SNN_FINE_TUNE and not use_best_config,
                use_best_temporal_config=use_best_config,
                temporal_spike_encoding=use_temporal_snn and TEMPORAL_SPIKE_ENCODING,
                encoding_steps=ENCODING_STEPS,
            )
        )
        snn_label = f"{task_name} Temporal SNN" if use_temporal_snn else f"{task_name} Tuned SNN"

    print(f"{task_name} SNN model trained")
    evaluate_classification(snn_y_test, snn_y_pred, snn_label, num_classes=num_classes)

    results: dict = {
        "snn": {
            "y_test": snn_y_test,
            "y_pred": snn_y_pred,
            "acc": snn_acc,
            "macro_f1": snn_macro_f1,
            "params": snn_params,
        },
    }
    if run_classical_models and y_test is not None and y_pred is not None:
        results["baseline"] = {
            "y_test": y_test,
            "y_pred": y_pred,
            "acc": acc,
            "macro_f1": baseline_macro_f1,
            "params": baseline_params,
        }

    return acc, baseline_macro_f1, baseline_params, snn_acc, snn_macro_f1, snn_params, results


def _run_temporal_window_optimization(
    X_eeg: np.ndarray,
    y_multi: np.ndarray,
    window_options: List[int],
) -> Tuple[Optional[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Step 31: train best Temporal SNN config for each window count; pick best by macro F1.

    X_eeg: (trials, channels, samples) — all channels, before classical channel selection.
    """
    print("\n=== Temporal window optimization (Step 31) ===")
    print("Temporal window options:", window_options)
    print("Skipping grid search, classical models, and temporal spike encoding")

    run_results: List[Dict[str, Any]] = []
    best_macro_f1 = -1.0
    best_entry: Optional[Dict[str, Any]] = None
    best_multi_results: Optional[Dict[str, Any]] = None

    for num_windows in window_options:
        print(f"\n--- Temporal windows: {num_windows} ---")
        X_temporal = extract_temporal_window_snn_features(
            X_eeg,
            num_windows=num_windows,
            use_frontal_asymmetry=USE_FRONTAL_ASYMMETRY_FEATURES,
        )
        print(f"Number of windows: {num_windows}")
        print(f"Feature shape: {X_temporal.shape}")

        _, _, snn_y_test, snn_y_pred, snn_acc, snn_macro_f1, snn_params = train_tuned_snn_model(
            X_temporal,
            y_multi,
            temporal=True,
            use_best_temporal_config=True,
            temporal_spike_encoding=False,
        )
        snn_params = {**snn_params, "num_windows": num_windows}
        evaluate_classification(
            snn_y_test,
            snn_y_pred,
            f"Multi-Emotion Temporal SNN ({num_windows} windows)",
            num_classes=4,
        )
        print(f"Accuracy: {snn_acc:.4f}")
        print(f"Macro F1: {snn_macro_f1:.4f}")

        entry = {
            "num_windows": num_windows,
            "feature_shape": tuple(X_temporal.shape),
            "accuracy": snn_acc,
            "macro_f1": snn_macro_f1,
            "params": snn_params,
        }
        run_results.append(entry)

        if snn_macro_f1 > best_macro_f1:
            best_macro_f1 = snn_macro_f1
            best_entry = entry
            best_multi_results = {
                "snn": {
                    "y_test": snn_y_test,
                    "y_pred": snn_y_pred,
                    "acc": snn_acc,
                    "macro_f1": snn_macro_f1,
                    "params": snn_params,
                },
            }

    if best_entry is not None:
        print("\n=== Best temporal window count (Step 31) ===")
        print("Selected by Macro F1")
        print(f"Number of windows: {best_entry['num_windows']}")
        print(f"Feature shape: {best_entry['feature_shape']}")
        print(f"Accuracy: {best_entry['accuracy']:.4f}")
        print(f"Macro F1: {best_entry['macro_f1']:.4f}")
        print("Params:", best_entry["params"])

    return best_multi_results, run_results


def main():
    if RUN_SEED_EXPERIMENT:
        print("SEED data directory:", SEED_DATA_DIR)
        try:
            if RUN_SEED_SUBJECT_SHIFT_STUDY:
                if SEED_SNN_MODE != "cnn_snn":
                    print(
                        "WARNING: RUN_SEED_SUBJECT_SHIFT_STUDY requires "
                        "SEED_SNN_MODE='cnn_snn'; skipping study."
                    )
                else:
                    run_seed_subject_shift_study(
                        data_dir=SEED_DATA_DIR,
                        split_mode=SEED_SPLIT_MODE,
                        cnn_snn_num_steps=CNN_SNN_NUM_STEPS,
                        fast_mode=SEED_SUBJECT_SHIFT_FAST,
                    )
            else:
                run_seed_experiment(
                    data_dir=SEED_DATA_DIR,
                    split_mode=SEED_SPLIT_MODE,
                    normalization_mode=SEED_NORMALIZATION_MODE,
                    snn_mode=SEED_SNN_MODE,
                    snn_fast_grid=SEED_SNN_FAST_GRID,
                    cnn_snn_fast_grid=SEED_CNN_SNN_FAST_GRID,
                    cnn_snn_num_steps=CNN_SNN_NUM_STEPS,
                )
        except FileNotFoundError as exc:
            print(f"SEED experiment skipped: {exc}")
        except ValueError as exc:
            print(f"SEED experiment error: {exc}")

        if RUN_SEED_ONLY:
            print("RUN_SEED_ONLY enabled — skipping DEAP pipeline")
            return

    folder = "data/raw"
    if not os.path.exists(folder):
        if RUN_SEED_EXPERIMENT:
            print("\nDEAP pipeline skipped (no data/raw folder).")
            return
        print("No DEAP .dat files found. Please place s01.dat ... s32.dat inside data/raw/")
        return

    dat_files = sorted(f for f in os.listdir(folder) if f.startswith("s") and f.endswith(".dat"))
    if len(dat_files) == 0:
        if RUN_SEED_EXPERIMENT:
            print("\nDEAP pipeline skipped (no s*.dat files in data/raw).")
            return
        print("No DEAP .dat files found. Please place s01.dat ... s32.dat inside data/raw/")
        return

    max_subjects = MAX_SUBJECTS if FAST_TEST_MODE else None
    if FAST_TEST_MODE:
        print("FAST_TEST_MODE enabled")
        print(f"Number of subject files used: {min(MAX_SUBJECTS, len(dat_files))}")

    X, y = load_all_deap_files(folder, max_subjects=max_subjects)
    X_filtered = bandpass_filter(X)
    subject_ids = _build_subject_ids(X_filtered.shape[0])

    if RUN_SNN_RESEARCH_EXPERIMENTS:
        X_normalized = None
        X_eeg_for_temporal = None
        X_temporal_snn = None
        print("Preprocessing: bandpass filter applied; normalization deferred per experiment")
        print("Filtered data shape:", X_filtered.shape)
    else:
        X_normalized = normalize_with_mode(
            X_filtered,
            NORMALIZATION_MODE,
            subject_ids=subject_ids,
        )
        print("Preprocessing completed")
        print_normalization_mode(NORMALIZATION_MODE)
        print("Processed data shape:", X_normalized.shape)

        X_eeg_for_temporal = X_normalized.copy()

        X_temporal_snn = None
        if USE_TEMPORAL_SNN_FEATURES and not RUN_TEMPORAL_WINDOW_OPTIMIZATION:
            X_temporal_snn = _prepare_temporal_snn_input(
                X_eeg_for_temporal,
                use_eeg_only=SNN_USE_EEG_ONLY_CHANNELS,
                feature_type=TEMPORAL_FEATURE_TYPE,
                use_frontal_asymmetry=USE_FRONTAL_ASYMMETRY_FEATURES,
            )
            if USE_FRONTAL_ASYMMETRY_FEATURES:
                print_frontal_asymmetry_feature_info(
                    X_temporal_snn,
                    num_windows=TEMPORAL_NUM_WINDOWS,
                )
            else:
                print_temporal_snn_feature_info(X_temporal_snn, num_windows=TEMPORAL_NUM_WINDOWS)
            if TEMPORAL_SPIKE_ENCODING:
                print_temporal_spike_encoding_info(
                    X_temporal_snn.shape[0],
                    X_temporal_snn.shape[1],
                    X_temporal_snn.shape[2],
                    encoding_steps=ENCODING_STEPS,
                )

    X_features = None
    if not RUN_SNN_RESEARCH_EXPERIMENTS:
        X_normalized, selected_channels = select_channels(
            X_normalized,
            CHANNEL_SELECTION_MODE,
            enabled=USE_CHANNEL_SELECTION,
        )
        print_channel_selection_info(
            CHANNEL_SELECTION_MODE,
            selected_channels,
            enabled=USE_CHANNEL_SELECTION,
        )
        if USE_CHANNEL_SELECTION:
            print("Data shape after channel selection:", X_normalized.shape)

        print_feature_mode_comparison(
            use_frequency_features=USE_FREQUENCY_FEATURES,
            use_differential_entropy=USE_DIFFERENTIAL_ENTROPY,
            use_combined_stat_de=USE_COMBINED_STAT_DE_FEATURES,
        )

        X_features = extract_features(
            X_normalized,
            use_frequency_features=USE_FREQUENCY_FEATURES,
            use_differential_entropy=USE_DIFFERENTIAL_ENTROPY,
            use_combined_stat_de=USE_COMBINED_STAT_DE_FEATURES,
        )
        active_mode = get_feature_mode_name(
            use_frequency_features=USE_FREQUENCY_FEATURES,
            use_differential_entropy=USE_DIFFERENTIAL_ENTROPY,
            use_combined_stat_de=USE_COMBINED_STAT_DE_FEATURES,
        )
        mode_info = FEATURE_MODES[active_mode]
        n_channels = X_normalized.shape[1]
        expected_size = get_expected_feature_size(active_mode, n_channels)
        print(f"Feature type: {mode_info['label']}")
        print("Feature shape:", X_features.shape)
        print(f"Expected feature size: {expected_size}")

        if REMOVE_CONSTANT_FEATURES:
            original_shape = X_features.shape
            X_features, n_removed = remove_constant_features(X_features, threshold=0.0)
            print(f"Original feature shape: {original_shape}")
            print(f"Cleaned feature shape: {X_features.shape}")
            print(f"Removed constant features: {n_removed}")

    # Binary labels (legacy, kept for comparison)
    y_binary = (y[:, 1] > 5).astype(int)
    print("Binary arousal labels created")
    print("y_binary shape:", y_binary.shape)

    # Multi-emotion labels (Step 13 / Step 22 / Step 30)
    if USE_AMBIGUOUS_SAMPLE_FILTER:
        keep_mask, y_multi_full = create_clear_multi_emotion_labels(
            y,
            low_threshold=LOW_THRESHOLD,
            high_threshold=HIGH_THRESHOLD,
        )
        print_ambiguous_sample_filter_summary(
            keep_mask,
            y_multi_full,
            low_threshold=LOW_THRESHOLD,
            high_threshold=HIGH_THRESHOLD,
        )
        y_multi = y_multi_full[keep_mask]
        if X_features is not None:
            X_features, y_binary = subset_arrays_by_mask(keep_mask, X_features, y_binary)
        else:
            y_binary = y_binary[keep_mask]
        if X_temporal_snn is not None:
            (X_temporal_snn,) = subset_arrays_by_mask(keep_mask, X_temporal_snn)
        if X_eeg_for_temporal is not None:
            X_eeg_for_temporal = X_eeg_for_temporal[keep_mask]
        if RUN_SNN_RESEARCH_EXPERIMENTS:
            X_filtered = X_filtered[keep_mask]
            subject_ids = subject_ids[keep_mask]
        y = y[keep_mask]
        print("Multi-emotion labels created (clear samples only)")
        print("y_multi shape:", y_multi.shape)
    else:
        compare_label_strategies(y, num_classes=4)
        y_multi = create_multi_emotion_labels(y, strategy=MULTI_LABEL_STRATEGY, verbose=True)
        print("Multi-emotion labels created")
        print("y_multi shape:", y_multi.shape)
        print("Class distribution (selected strategy):")
        print_class_distribution(y_multi, EMOTION_LABELS, num_classes=4)

    empty_classes = get_empty_classes(y_multi, num_classes=4)
    skip_multi_emotion = len(empty_classes) > 0
    if skip_multi_emotion:
        empty_names = [EMOTION_LABELS[c] for c in empty_classes]
        print(
            f"WARNING: Multi-emotion labels have empty classes "
            f"{empty_classes} ({empty_names}). Skipping multi-emotion training."
        )

    if not RUN_CLASSICAL_MODELS and not RUN_SNN_RESEARCH_EXPERIMENTS:
        print("Skipping classical models")

    research_results: List[Dict[str, Any]] = []
    if not skip_multi_emotion and RUN_SNN_RESEARCH_EXPERIMENTS:
        research_results = _run_snn_research_experiments(
            X_filtered,
            y_multi,
            subject_ids,
        )

    binary_results = None
    if RUN_BINARY_CLASSIFICATION:
        print("\n--- Binary classification (Calm vs Excited) ---")
        _, _, _, _, _, _, binary_results = _run_classification_pipeline(
            X_features,
            y_binary,
            X_snn_features=X_temporal_snn,
            task_name="Binary",
            num_classes=2,
            run_classical_models=RUN_CLASSICAL_MODELS,
        )

    multi_results = None
    window_opt_results: List[Dict[str, Any]] = []
    acc = baseline_macro_f1 = snn_acc = snn_macro_f1 = rf_acc = rf_macro_f1 = 0.0
    baseline_params = snn_params = rf_params = {}

    if not skip_multi_emotion and RUN_TEMPORAL_WINDOW_OPTIMIZATION:
        multi_results, window_opt_results = _run_temporal_window_optimization(
            X_eeg_for_temporal,
            y_multi,
            TEMPORAL_WINDOW_OPTIONS,
        )
        if multi_results is not None:
            snn_acc = multi_results["snn"]["acc"]
            snn_macro_f1 = multi_results["snn"]["macro_f1"]
            snn_params = multi_results["snn"]["params"]
    elif not skip_multi_emotion and not RUN_SNN_RESEARCH_EXPERIMENTS:
        print("\n--- Multi-emotion classification (Valence-Arousal) ---")
        acc, baseline_macro_f1, baseline_params, snn_acc, snn_macro_f1, snn_params, multi_results = (
            _run_classification_pipeline(
                X_features,
                y_multi,
                X_snn_features=X_temporal_snn,
                task_name="Multi-Emotion",
                num_classes=4,
                run_classical_models=RUN_CLASSICAL_MODELS,
            )
        )

        if RUN_CLASSICAL_MODELS:
            print("\n--- Random Forest with feature selection (Multi-Emotion) ---")
            rf_model, _, rf_y_test, rf_y_pred, rf_acc, rf_macro_f1, rf_params = (
                train_random_forest_model(X_features, y_multi)
            )
            evaluate_classification(
                rf_y_test, rf_y_pred, "Multi-Emotion Random Forest", num_classes=4
            )

        print("\n=== Comparison summary ===")
        print("Multi-Emotion label strategy:", MULTI_LABEL_STRATEGY)
        if RUN_CLASSICAL_MODELS:
            print("Multi-Emotion Baseline Accuracy:", acc)
            print("Multi-Emotion Baseline Macro F1:", baseline_macro_f1)
            print("Multi-Emotion Baseline Params:", baseline_params)
        print("Active SNN mode:", snn_params.get("mode", "unknown"))
        print("Multi-Emotion SNN Accuracy:", snn_acc)
        print("Multi-Emotion SNN Macro F1:", snn_macro_f1)
        print("Multi-Emotion SNN Params:", snn_params)
        if RUN_CLASSICAL_MODELS:
            print("Multi-Emotion Random Forest Accuracy:", rf_acc)
            print("Multi-Emotion Random Forest Macro F1:", rf_macro_f1)
            print("Multi-Emotion Random Forest Params:", rf_params)

        print("\n=== Final Model Comparison (Multi-Emotion) ===")
        if RUN_CLASSICAL_MODELS:
            print("Logistic Regression")
            print(f"  Accuracy: {acc:.4f}")
            print(f"  Macro F1: {baseline_macro_f1:.4f}")
        print("SNN")
        print(f"  Accuracy: {snn_acc:.4f}")
        print(f"  Macro F1: {snn_macro_f1:.4f}")
        if RUN_CLASSICAL_MODELS:
            print("Random Forest")
            print(f"  Accuracy: {rf_acc:.4f}")
            print(f"  Macro F1: {rf_macro_f1:.4f}")

    if (
        multi_results is not None
        and RUN_CLASSICAL_MODELS
        and binary_results is not None
        and not RUN_TEMPORAL_WINDOW_OPTIMIZATION
    ):
        generate_all_figures(
            binary_results,
            multi_results,
            binary_label_names=[BINARY_LABELS[0], BINARY_LABELS[1]],
            multi_label_names=[EMOTION_LABELS[i] for i in range(4)],
        )
        print("Visualization completed")
    elif multi_results is not None:
        print("Visualization skipped (classical models disabled or binary task skipped)")
    elif RUN_SNN_RESEARCH_EXPERIMENTS and research_results:
        print("Visualization skipped (SNN research experiments mode)")
    else:
        print("Visualization skipped (multi-emotion training skipped due to empty classes)")

    results = []
    if binary_results is not None:
        if RUN_CLASSICAL_MODELS and "baseline" in binary_results:
            results.append(
                {
                    "task": "Binary Classification",
                    "model": "Baseline Logistic Regression",
                    "accuracy": binary_results["baseline"]["acc"],
                    "macro_f1": binary_results["baseline"]["macro_f1"],
                    "best_params": binary_results["baseline"]["params"],
                    "notes": "Best binary baseline model",
                }
            )
        results.append(
            {
                "task": "Binary Classification",
                "model": "Tuned SNN",
                "accuracy": binary_results["snn"]["acc"],
                "macro_f1": binary_results["snn"]["macro_f1"],
                "best_params": binary_results["snn"]["params"],
                "notes": "Best tuned binary SNN model",
            }
        )
    if research_results:
        for entry in research_results:
            results.append(
                {
                    "task": "Multi-Emotion Classification",
                    "model": "Temporal SNN (Step 33 research)",
                    "accuracy": entry["accuracy"],
                    "macro_f1": entry["macro_f1"],
                    "best_params": entry.get("params", {}),
                    "notes": (
                        f"norm={entry['normalization']} "
                        f"feature={entry['feature_type']} "
                        f"shape={entry['feature_shape']} "
                        f"eeg_only={SNN_USE_EEG_ONLY_CHANNELS}"
                    ),
                }
            )
    elif multi_results is not None:
        if RUN_TEMPORAL_WINDOW_OPTIMIZATION:
            for entry in window_opt_results:
                results.append(
                    {
                        "task": "Multi-Emotion Classification",
                        "model": "Temporal SNN (window optimization)",
                        "accuracy": entry["accuracy"],
                        "macro_f1": entry["macro_f1"],
                        "best_params": entry["params"],
                        "notes": (
                            f"Step 31 windows={entry['num_windows']} "
                            f"shape={entry['feature_shape']} "
                            f"strategy={MULTI_LABEL_STRATEGY}"
                        ),
                    }
                )
        elif RUN_CLASSICAL_MODELS and "baseline" in multi_results:
            results.append(
                {
                    "task": "Multi-Emotion Classification",
                    "model": "Baseline Logistic Regression",
                    "accuracy": multi_results["baseline"]["acc"],
                    "macro_f1": multi_results["baseline"]["macro_f1"],
                    "best_params": multi_results["baseline"]["params"],
                    "notes": f"4-class Valence-Arousal baseline (strategy={MULTI_LABEL_STRATEGY})",
                }
            )
        if not RUN_TEMPORAL_WINDOW_OPTIMIZATION:
            results.append(
                {
                    "task": "Multi-Emotion Classification",
                    "model": "Tuned SNN",
                    "accuracy": multi_results["snn"]["acc"],
                    "macro_f1": multi_results["snn"]["macro_f1"],
                    "best_params": multi_results["snn"]["params"],
                    "notes": f"4-class Valence-Arousal tuned SNN (strategy={MULTI_LABEL_STRATEGY})",
                }
            )

    export_results_summary(results)
    print("Results summary exported")
    print("Saved to results/metrics/results_summary.csv")
    print("Saved to results/metrics/results_summary.json")


if __name__ == "__main__":
    main()
