from src.load_data import load_deap_file
from src.preprocessing import bandpass_filter, normalize
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


if __name__ == "__main__":
    main()

