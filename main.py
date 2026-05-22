from src.load_data import load_all_deap_files
from src.preprocessing import bandpass_filter, normalize
from src.features import extract_features
from src.baseline_model import train_baseline_model
from src.snn_model import train_snn_model
from src.evaluate import evaluate_classification
import os


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

    y_binary = (y[:, 1] > 5).astype(int)
    print("Labels created")
    print("y_binary shape:", y_binary.shape)

    model, X_test, y_test, y_pred, acc, baseline_macro_f1, baseline_params = (
        train_baseline_model(X_features, y_binary)
    )
    print("Tuned baseline model trained")
    evaluate_classification(y_test, y_pred, "Baseline Logistic Regression")

    snn_model, snn_X_test, snn_y_test, snn_y_pred, snn_acc, snn_macro_f1, snn_params = (
        train_snn_model(X_features, y_binary)
    )
    print("Spike-encoded SNN model trained")
    print("Spike-encoded SNN accuracy:", snn_acc)
    print("Spike-encoded SNN macro F1:", snn_macro_f1)
    evaluate_classification(snn_y_test, snn_y_pred, "Spike-encoded SNN")

    print("\n=== Comparison summary ===")
    print("Best Baseline Accuracy:", acc)
    print("Best Baseline Macro F1:", baseline_macro_f1)
    print("Best Baseline Params:", baseline_params)
    print("Best SNN Accuracy:", snn_acc)
    print("Best SNN Macro F1:", snn_macro_f1)
    print("Best SNN Params:", snn_params)


if __name__ == "__main__":
    main()

