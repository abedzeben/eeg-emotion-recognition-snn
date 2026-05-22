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

    model, X_test, y_test, y_pred, acc = train_baseline_model(X_features, y_binary)
    print("Baseline model trained")
    print("Baseline accuracy:", acc)
    evaluate_classification(y_test, y_pred, "Baseline Logistic Regression")

    snn_model, snn_X_test, snn_y_test, snn_y_pred, snn_acc = train_snn_model(
        X_features, y_binary
    )
    print("Improved SNN model trained")
    print("Improved SNN accuracy:", snn_acc)
    evaluate_classification(snn_y_test, snn_y_pred, "Improved SNN")

    print("\n=== Comparison summary ===")
    print("Baseline accuracy:", acc)
    print("SNN accuracy:", snn_acc)


if __name__ == "__main__":
    main()

