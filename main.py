from src.load_data import load_deap_file
from src.preprocessing import bandpass_filter, normalize
from src.features import extract_features
import os


def main():
    path = "data/raw/s01.dat"
    if not os.path.exists(path):
        print("DEAP file not found. Please place s01.dat inside data/raw/")
        return

    X, y = load_deap_file(path)
    X_filtered = bandpass_filter(X)
    X_normalized = normalize(X_filtered)

    print("Preprocessing completed")
    print("Processed data shape:", X_normalized.shape)

    X_features = extract_features(X_normalized)
    print("Feature extraction completed")
    print("X_features shape:", X_features.shape)

    y_binary = (y[:, 1] > 5).astype(int)
    print("Labels created")
    print("y_binary shape:", y_binary.shape)


if __name__ == "__main__":
    main()

