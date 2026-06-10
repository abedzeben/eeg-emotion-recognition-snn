from src.load_data import load_all_deap_files
from src.preprocessing import bandpass_filter, normalize
from src.features import extract_features, remove_constant_features
from src.baseline_model import train_baseline_model
from src.random_forest_model import train_random_forest_model
from src.snn_model import train_tuned_snn_model, train_spike_encoded_snn_model
from src.evaluate import evaluate_classification
from src.labels import (
    BINARY_LABELS,
    compare_label_strategies,
    create_multi_emotion_labels,
    get_empty_classes,
    print_class_distribution,
    EMOTION_LABELS,
)
from src.visualize import generate_all_figures
from src.results_export import export_results_summary
import os

# SNN mode: False = Step 11 tuned SNN (default), True = Step 12 spike-encoded SNN (experimental)
USE_SPIKE_ENCODING = False

# Set True to also run the legacy binary arousal pipeline (Calm vs Excited)
RUN_BINARY_CLASSIFICATION = False

# Step 19: add Welch PSD band-power features on top of extended statistical features
USE_FREQUENCY_FEATURES = False

# Step 21: remove constant / near-constant features before training
REMOVE_CONSTANT_FEATURES = True

# Step 22: multi-emotion Valence-Arousal threshold strategy
MULTI_LABEL_STRATEGY = "median"


def _run_classification_pipeline(X_features, y, *, task_name: str, num_classes: int):
    """Train baseline + SNN and evaluate for the given label vector."""
    model, X_test, y_test, y_pred, acc, baseline_macro_f1, baseline_params = train_baseline_model(
        X_features, y
    )
    print(f"{task_name} baseline model trained")
    evaluate_classification(y_test, y_pred, f"{task_name} Baseline", num_classes=num_classes)

    if USE_SPIKE_ENCODING:
        print("Running Spike-Encoded SNN (Step 12)")
        snn_model, snn_X_test, snn_y_test, snn_y_pred, snn_acc, snn_macro_f1, snn_params = (
            train_spike_encoded_snn_model(X_features, y)
        )
        snn_label = f"{task_name} Spike-encoded SNN"
    else:
        print("Running Tuned SNN (Step 11)")
        snn_model, snn_X_test, snn_y_test, snn_y_pred, snn_acc, snn_macro_f1, snn_params = (
            train_tuned_snn_model(X_features, y)
        )
        snn_label = f"{task_name} Tuned SNN"

    print(f"{task_name} SNN model trained")
    evaluate_classification(snn_y_test, snn_y_pred, snn_label, num_classes=num_classes)

    results = {
        "baseline": {
            "y_test": y_test,
            "y_pred": y_pred,
            "acc": acc,
            "macro_f1": baseline_macro_f1,
            "params": baseline_params,
        },
        "snn": {
            "y_test": snn_y_test,
            "y_pred": snn_y_pred,
            "acc": snn_acc,
            "macro_f1": snn_macro_f1,
            "params": snn_params,
        },
    }

    return acc, baseline_macro_f1, baseline_params, snn_acc, snn_macro_f1, snn_params, results


def main():
    folder = "data/raw"
    if not os.path.exists(folder):
        print("No DEAP .dat files found. Please place s01.dat ... s32.dat inside data/raw/")
        return

    dat_files = [f for f in os.listdir(folder) if f.startswith("s") and f.endswith(".dat")]
    if len(dat_files) == 0:
        print("No DEAP .dat files found. Please place s01.dat ... s32.dat inside data/raw/")
        return

    X, y = load_all_deap_files(folder)
    X_filtered = bandpass_filter(X)
    X_normalized = normalize(X_filtered)

    print("Preprocessing completed")
    print("Processed data shape:", X_normalized.shape)

    X_features = extract_features(X_normalized, use_frequency_features=USE_FREQUENCY_FEATURES)
    if USE_FREQUENCY_FEATURES:
        print("Frequency feature extraction enabled")
        print("X_features shape:", X_features.shape)
        print("Expected feature size: 440")
    else:
        print("Improved feature extraction completed")
        print("X_features shape:", X_features.shape)

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

    # Multi-emotion labels (Step 13 / Step 22 strategies)
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
            f"WARNING: Selected strategy '{MULTI_LABEL_STRATEGY}' has empty classes "
            f"{empty_classes} ({empty_names}). Skipping multi-emotion training."
        )

    print("\n--- Binary classification (Calm vs Excited) ---")
    _, _, _, _, _, _, binary_results = _run_classification_pipeline(
        X_features, y_binary, task_name="Binary", num_classes=2
    )

    multi_results = None
    acc = baseline_macro_f1 = snn_acc = snn_macro_f1 = rf_acc = rf_macro_f1 = 0.0
    baseline_params = snn_params = rf_params = {}

    if not skip_multi_emotion:
        print("\n--- Multi-emotion classification (Valence-Arousal) ---")
        acc, baseline_macro_f1, baseline_params, snn_acc, snn_macro_f1, snn_params, multi_results = (
            _run_classification_pipeline(
                X_features, y_multi, task_name="Multi-Emotion", num_classes=4
            )
        )

        print("\n--- Random Forest with feature selection (Multi-Emotion) ---")
        rf_model, _, rf_y_test, rf_y_pred, rf_acc, rf_macro_f1, rf_params = (
            train_random_forest_model(X_features, y_multi)
        )
        evaluate_classification(
            rf_y_test, rf_y_pred, "Multi-Emotion Random Forest", num_classes=4
        )

        print("\n=== Comparison summary ===")
        print("Multi-Emotion label strategy:", MULTI_LABEL_STRATEGY)
        print("Multi-Emotion Baseline Accuracy:", acc)
        print("Multi-Emotion Baseline Macro F1:", baseline_macro_f1)
        print("Multi-Emotion Baseline Params:", baseline_params)
        print("Active SNN mode:", snn_params.get("mode", "unknown"))
        print("Multi-Emotion SNN Accuracy:", snn_acc)
        print("Multi-Emotion SNN Macro F1:", snn_macro_f1)
        print("Multi-Emotion SNN Params:", snn_params)
        print("Multi-Emotion Random Forest Accuracy:", rf_acc)
        print("Multi-Emotion Random Forest Macro F1:", rf_macro_f1)
        print("Multi-Emotion Random Forest Params:", rf_params)

        print("\n=== Final Model Comparison (Multi-Emotion) ===")
        print("Logistic Regression")
        print(f"  Accuracy: {acc:.4f}")
        print(f"  Macro F1: {baseline_macro_f1:.4f}")
        print("SNN")
        print(f"  Accuracy: {snn_acc:.4f}")
        print(f"  Macro F1: {snn_macro_f1:.4f}")
        print("Random Forest")
        print(f"  Accuracy: {rf_acc:.4f}")
        print(f"  Macro F1: {rf_macro_f1:.4f}")

    if multi_results is not None:
        generate_all_figures(
            binary_results,
            multi_results,
            binary_label_names=[BINARY_LABELS[0], BINARY_LABELS[1]],
            multi_label_names=[EMOTION_LABELS[i] for i in range(4)],
        )
        print("Visualization completed")
    else:
        print("Visualization skipped (multi-emotion training skipped due to empty classes)")

    results = [
        {
            "task": "Binary Classification",
            "model": "Baseline Logistic Regression",
            "accuracy": binary_results["baseline"]["acc"],
            "macro_f1": binary_results["baseline"]["macro_f1"],
            "best_params": binary_results["baseline"]["params"],
            "notes": "Best binary baseline model",
        },
        {
            "task": "Binary Classification",
            "model": "Tuned SNN",
            "accuracy": binary_results["snn"]["acc"],
            "macro_f1": binary_results["snn"]["macro_f1"],
            "best_params": binary_results["snn"]["params"],
            "notes": "Best tuned binary SNN model",
        },
    ]
    if multi_results is not None:
        results.extend(
            [
                {
                    "task": "Multi-Emotion Classification",
                    "model": "Baseline Logistic Regression",
                    "accuracy": multi_results["baseline"]["acc"],
                    "macro_f1": multi_results["baseline"]["macro_f1"],
                    "best_params": multi_results["baseline"]["params"],
                    "notes": f"4-class Valence-Arousal baseline (strategy={MULTI_LABEL_STRATEGY})",
                },
                {
                    "task": "Multi-Emotion Classification",
                    "model": "Tuned SNN",
                    "accuracy": multi_results["snn"]["acc"],
                    "macro_f1": multi_results["snn"]["macro_f1"],
                    "best_params": multi_results["snn"]["params"],
                    "notes": f"4-class Valence-Arousal tuned SNN (strategy={MULTI_LABEL_STRATEGY})",
                },
            ]
        )

    export_results_summary(results)
    print("Results summary exported")
    print("Saved to results/metrics/results_summary.csv")
    print("Saved to results/metrics/results_summary.json")


if __name__ == "__main__":
    main()
