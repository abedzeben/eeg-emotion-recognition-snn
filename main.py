from src.load_data import load_all_deap_files
from src.preprocessing import bandpass_filter, normalize
from src.features import extract_features
from src.baseline_model import train_baseline_model
from src.snn_model import train_tuned_snn_model, train_spike_encoded_snn_model
from src.evaluate import evaluate_classification
from src.labels import create_multi_emotion_labels, print_class_distribution, EMOTION_LABELS
import os

# SNN mode: False = Step 11 tuned SNN (default), True = Step 12 spike-encoded SNN (experimental)
USE_SPIKE_ENCODING = False

# Set True to also run the legacy binary arousal pipeline (Calm vs Excited)
RUN_BINARY_CLASSIFICATION = False


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

    return acc, baseline_macro_f1, baseline_params, snn_acc, snn_macro_f1, snn_params


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

    X_features = extract_features(X_normalized)
    print("Improved feature extraction completed")
    print("X_features shape:", X_features.shape)

    # Binary labels (legacy, kept for comparison)
    y_binary = (y[:, 1] > 5).astype(int)
    print("Binary arousal labels created")
    print("y_binary shape:", y_binary.shape)

    # Multi-emotion labels (Step 13 primary pipeline)
    y_multi = create_multi_emotion_labels(y)
    print("Multi-emotion labels created")
    print("y_multi shape:", y_multi.shape)
    print("Class distribution:")
    print_class_distribution(y_multi, EMOTION_LABELS, num_classes=4)

    if RUN_BINARY_CLASSIFICATION:
        print("\n--- Binary classification (legacy) ---")
        _run_classification_pipeline(
            X_features, y_binary, task_name="Binary", num_classes=2
        )

    print("\n--- Multi-emotion classification (Valence-Arousal) ---")
    acc, baseline_macro_f1, baseline_params, snn_acc, snn_macro_f1, snn_params = (
        _run_classification_pipeline(
            X_features, y_multi, task_name="Multi-Emotion", num_classes=4
        )
    )

    print("\n=== Comparison summary ===")
    print("Multi-Emotion Baseline Accuracy:", acc)
    print("Multi-Emotion Baseline Macro F1:", baseline_macro_f1)
    print("Multi-Emotion Baseline Params:", baseline_params)
    print("Active SNN mode:", snn_params.get("mode", "unknown"))
    print("Multi-Emotion SNN Accuracy:", snn_acc)
    print("Multi-Emotion SNN Macro F1:", snn_macro_f1)
    print("Multi-Emotion SNN Params:", snn_params)


if __name__ == "__main__":
    main()
