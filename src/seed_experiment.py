from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.snn_model import train_seed_snn_model

SEED_LABEL_NAMES: Dict[int, str] = {
    0: "Negative",
    1: "Neutral",
    2: "Positive",
}

SEED_SPLIT_MODES = ("trial", "subject")
SeedSplitMode = Literal["trial", "subject"]
SEED_NORMALIZATION_MODES = ("global", "train_only_standard")
SeedNormalizationMode = Literal["global", "train_only_standard"]
SEED_SNN_MODES = ("simple", "strong", "cnn_snn")
SeedSnnMode = Literal["simple", "strong", "cnn_snn"]

SEED_TRAIN_SUBJECTS = list(range(0, 12))
SEED_TEST_SUBJECTS = list(range(12, 15))

DEFAULT_SEED_FILES = {
    "X": "DatasetCaricatoNoImage.npz",
    "y": "LabelsNoImage.npz",
    "subjects": "SubjectsNoImage.npz",
}


def load_seed_dataset(
    data_dir: Union[str, Path],
    *,
    x_file: str = DEFAULT_SEED_FILES["X"],
    y_file: str = DEFAULT_SEED_FILES["y"],
    subjects_file: str = DEFAULT_SEED_FILES["subjects"],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load SEED NPZ files (arr_0 key in each)."""
    data_dir = Path(data_dir)
    x_path = data_dir / x_file
    y_path = data_dir / y_file
    subjects_path = data_dir / subjects_file

    for path in (x_path, y_path, subjects_path):
        if not path.exists():
            raise FileNotFoundError(
                f"SEED file not found: {path}. "
                f"Place NPZ files in {data_dir}/"
            )

    X = np.load(x_path)["arr_0"].astype(np.float32)
    y = np.load(y_path)["arr_0"].astype(np.int64)
    subjects = np.load(subjects_path)["arr_0"].astype(np.int64)

    if X.ndim != 3:
        raise ValueError(f"Expected X shape (samples, bands, channels), got {X.shape}")
    if y.ndim != 1 or subjects.ndim != 1:
        raise ValueError("Expected y and subjects to be 1D arrays")
    if not (X.shape[0] == y.shape[0] == subjects.shape[0]):
        raise ValueError(
            f"Sample count mismatch: X={X.shape[0]}, y={y.shape[0]}, "
            f"subjects={subjects.shape[0]}"
        )

    return X, y, subjects


def print_seed_dataset_summary(
    X: np.ndarray,
    y: np.ndarray,
    subjects: np.ndarray,
) -> None:
    """Print SEED dataset shapes and class/subject distributions."""
    print("\n=== SEED dataset summary ===")
    print("X shape:", X.shape)
    print("y shape:", y.shape)
    print("subjects shape:", subjects.shape)
    print("Time steps (frequency bands):", X.shape[1])
    print("Features per step (EEG channels):", X.shape[2])

    unique_labels, label_counts = np.unique(y, return_counts=True)
    print("Unique labels:", unique_labels.tolist())
    print("Label counts:")
    for label, count in zip(unique_labels, label_counts):
        name = SEED_LABEL_NAMES.get(int(label), f"Class {label}")
        print(f"  {int(label)} ({name}): {int(count)}")

    unique_subjects = np.unique(subjects)
    print("Unique subjects:", unique_subjects.tolist())
    print("Samples per subject:")
    for subject in unique_subjects:
        count = int(np.sum(subjects == subject))
        print(f"  Subject {int(subject)}: {count}")


def split_seed_data(
    X: np.ndarray,
    y: np.ndarray,
    subjects: np.ndarray,
    *,
    split_mode: SeedSplitMode = "trial",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    """Split SEED data by random trial or by subject."""
    split_info: Dict[str, Any] = {"split_mode": split_mode}

    if split_mode == "trial":
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42,
            stratify=y,
        )
        split_info["test_size"] = 0.2
        split_info["random_state"] = 42
        split_info["train_samples"] = int(X_train.shape[0])
        split_info["test_samples"] = int(X_test.shape[0])
        return X_train, X_test, y_train, y_test, split_info

    if split_mode == "subject":
        train_mask = np.isin(subjects, SEED_TRAIN_SUBJECTS)
        test_mask = np.isin(subjects, SEED_TEST_SUBJECTS)
        if not np.any(test_mask):
            raise ValueError("No test samples found for subjects 12–14")
        if not np.any(train_mask):
            raise ValueError("No train samples found for subjects 0–11")

        X_train = X[train_mask]
        X_test = X[test_mask]
        y_train = y[train_mask]
        y_test = y[test_mask]
        split_info["train_subjects"] = SEED_TRAIN_SUBJECTS
        split_info["test_subjects"] = SEED_TEST_SUBJECTS
        split_info["train_samples"] = int(X_train.shape[0])
        split_info["test_samples"] = int(X_test.shape[0])
        print("Train subjects:", SEED_TRAIN_SUBJECTS)
        print("Test subjects:", SEED_TEST_SUBJECTS)
        return X_train, X_test, y_train, y_test, split_info

    raise ValueError(f"Unknown SEED split mode: {split_mode}")


def normalize_seed_features(
    X_train: np.ndarray,
    X_test: np.ndarray,
    *,
    mode: SeedNormalizationMode = "train_only_standard",
    X_full: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Normalize SEED 3D features.

    global: fit StandardScaler on full dataset (requires X_full).
    train_only_standard: fit on training data only.
    """
    n_steps = X_train.shape[1]
    n_features = X_train.shape[2]
    scaler = StandardScaler()

    if mode == "global":
        if X_full is None:
            raise ValueError("X_full is required for global SEED normalization")
        scaler.fit(X_full.reshape(-1, n_features))
    elif mode == "train_only_standard":
        scaler.fit(X_train.reshape(-1, n_features))
    else:
        raise ValueError(f"Unknown SEED normalization mode: {mode}")

    X_train_n = scaler.transform(X_train.reshape(-1, n_features)).reshape(
        X_train.shape[0], n_steps, n_features
    )
    X_test_n = scaler.transform(X_test.reshape(-1, n_features)).reshape(
        X_test.shape[0], n_steps, n_features
    )
    return X_train_n.astype(np.float32), X_test_n.astype(np.float32)


def train_seed_baseline(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
) -> Tuple[np.ndarray, float, float, Dict[str, Any]]:
    """
    Simple Logistic Regression baseline on flattened SEED features (no grid search).
    """
    X_train_flat = X_train.reshape(X_train.shape[0], -1)
    X_test_flat = X_test.reshape(X_test.shape[0], -1)

    clf = LogisticRegression(
        C=1.0,
        max_iter=1000,
        random_state=42,
    )
    clf.fit(X_train_flat, y_train)
    y_pred = clf.predict(X_test_flat)

    acc = float(accuracy_score(y_test, y_pred))
    macro_f1 = float(f1_score(y_test, y_pred, average="macro", zero_division=0))
    params = {"model": "LogisticRegression", "C": 1.0, "max_iter": 1000}
    return y_pred, acc, macro_f1, params


def evaluate_seed_predictions(
    y_test: np.ndarray,
    y_pred: np.ndarray,
    *,
    model_name: str,
    num_classes: int = 3,
) -> Dict[str, float]:
    """Print and return SEED classification metrics."""
    label_ids = list(range(num_classes))
    target_names = [SEED_LABEL_NAMES.get(i, f"Class {i}") for i in label_ids]

    acc = float(accuracy_score(y_test, y_pred))
    macro_f1 = float(f1_score(y_test, y_pred, average="macro", labels=label_ids, zero_division=0))
    weighted_f1 = float(
        f1_score(y_test, y_pred, average="weighted", labels=label_ids, zero_division=0)
    )

    print(f"\n=== {model_name} (SEED) ===")
    print("Accuracy:", f"{acc:.4f}")
    print("Macro F1:", f"{macro_f1:.4f}")
    print("Weighted F1:", f"{weighted_f1:.4f}")
    print("Confusion matrix:")
    print(confusion_matrix(y_test, y_pred, labels=label_ids))
    print("Classification report:")
    print(
        classification_report(
            y_test,
            y_pred,
            labels=label_ids,
            target_names=target_names,
            zero_division=0,
        )
    )

    return {
        "accuracy": acc,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
    }


def export_seed_results(
    results: List[Dict[str, Any]],
    output_dir: Union[str, Path] = "results/metrics",
) -> Tuple[Path, Path]:
    """Export SEED experiment summary to CSV and JSON."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / "seed_results_summary.csv"
    json_path = output_dir / "seed_results_summary.json"

    rows = []
    for entry in results:
        rows.append(
            {
                "dataset": "SEED",
                "model": entry["model"],
                "split_mode": entry.get("split_mode", ""),
                "normalization_mode": entry.get("normalization_mode", ""),
                "accuracy": float(entry["accuracy"]),
                "macro_f1": float(entry["macro_f1"]),
                "weighted_f1": float(entry.get("weighted_f1", 0.0)),
                "params": json.dumps(entry.get("params", {})),
                "notes": entry.get("notes", ""),
            }
        )

    pd.DataFrame(rows).to_csv(csv_path, index=False)

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    return csv_path, json_path


def run_seed_experiment(
    *,
    data_dir: Union[str, Path] = "data/seed",
    split_mode: SeedSplitMode = "trial",
    normalization_mode: SeedNormalizationMode = "train_only_standard",
    snn_mode: SeedSnnMode = "simple",
    snn_fast_grid: bool = True,
    cnn_snn_fast_grid: bool = True,
    cnn_snn_num_steps: int = 10,
) -> List[Dict[str, Any]]:
    """
    Step 34/35/36: full SEED SNN experiment with baseline comparison.
    """
    print("\n" + "=" * 60)
    print("SEED SNN Experiment (Step 34)")
    print("=" * 60)
    print("SEED_SPLIT_MODE:", split_mode)
    print("SEED_NORMALIZATION_MODE:", normalization_mode)
    print("SEED_SNN_MODE:", snn_mode)
    if snn_mode == "strong":
        print("SEED_SNN_FAST_GRID:", snn_fast_grid)
    if snn_mode == "cnn_snn":
        print("SEED_CNN_SNN_FAST_GRID:", cnn_snn_fast_grid)
        print("CNN_SNN_NUM_STEPS:", cnn_snn_num_steps)

    X, y, subjects = load_seed_dataset(data_dir)
    print_seed_dataset_summary(X, y, subjects)

    X_train, X_test, y_train, y_test, split_info = split_seed_data(
        X, y, subjects, split_mode=split_mode
    )
    print(f"\nTrain samples: {X_train.shape[0]}, Test samples: {X_test.shape[0]}")
    print("Temporal SNN input shape (train):", X_train.shape)

    X_train, X_test = normalize_seed_features(
        X_train,
        X_test,
        mode=normalization_mode,
        X_full=X if normalization_mode == "global" else None,
    )
    print("Normalized temporal feature shape (train):", X_train.shape)

    baseline_pred, baseline_acc, baseline_macro_f1, baseline_params = train_seed_baseline(
        X_train, X_test, y_train, y_test
    )
    baseline_metrics = evaluate_seed_predictions(
        y_test, baseline_pred, model_name="SEED Baseline (Logistic Regression)"
    )

    if snn_mode == "cnn_snn":
        from src.seed_cnn_snn import (
            export_cnn_snn_results,
            print_cnn_snn_seed_summary,
            train_cnn_snn_grid,
        )

        print("\n--- Training CNN-SNN Hybrid (Step 36) ---")
        print("Input shape:", X_train.shape, "→ CNN map (1, 5, 62)")
        print("Output classes: 3")
        snn_y_pred, grid_results, best_entry = train_cnn_snn_grid(
            X_train,
            y_train,
            X_test,
            y_test,
            fast_grid=cnn_snn_fast_grid,
            num_steps=cnn_snn_num_steps,
            num_classes=3,
        )
        snn_metrics = evaluate_seed_predictions(
            y_test, snn_y_pred, model_name="SEED CNN-SNN Hybrid (best config)"
        )
        print_cnn_snn_seed_summary(best_entry)

        results = [
            {
                "model": "Logistic Regression",
                "split_mode": split_mode,
                "normalization_mode": normalization_mode,
                "accuracy": baseline_metrics["accuracy"],
                "macro_f1": baseline_metrics["macro_f1"],
                "weighted_f1": baseline_metrics["weighted_f1"],
                "params": baseline_params,
                "notes": f"Flattened X shape {X.shape[1] * X.shape[2]}; split={split_info}",
            },
            {
                "model": "CNN-SNN Hybrid",
                "split_mode": split_mode,
                "normalization_mode": normalization_mode,
                "SEED_SNN_MODE": "cnn_snn",
                "accuracy": snn_metrics["accuracy"],
                "macro_f1": snn_metrics["macro_f1"],
                "weighted_f1": snn_metrics["weighted_f1"],
                "params": best_entry,
                "notes": (
                    f"Best of {len(grid_results)} configs; "
                    f"CNN map (1, {X.shape[1]}, {X.shape[2]}); split={split_info}"
                ),
            },
        ]

        csv_path, json_path = export_cnn_snn_results(grid_results, best_entry)
        print("\nCNN-SNN SEED results saved:")
        print(" ", csv_path)
        print(" ", json_path)
        return results

    if snn_mode == "strong":
        from src.seed_strong_snn import (
            export_strong_seed_results,
            print_strong_seed_snn_summary,
            train_strong_seed_snn_grid,
        )

        print("\n--- Training Strong SEED Temporal SNN (Step 35) ---")
        print("Input shape:", X_train.shape)
        print("Output classes: 3")
        snn_y_pred, grid_results, best_entry = train_strong_seed_snn_grid(
            X_train,
            y_train,
            X_test,
            y_test,
            fast_grid=snn_fast_grid,
            num_classes=3,
        )
        snn_metrics = evaluate_seed_predictions(
            y_test, snn_y_pred, model_name="SEED Strong Temporal SNN (best config)"
        )
        print_strong_seed_snn_summary(best_entry)

        results = [
            {
                "model": "Logistic Regression",
                "split_mode": split_mode,
                "normalization_mode": normalization_mode,
                "accuracy": baseline_metrics["accuracy"],
                "macro_f1": baseline_metrics["macro_f1"],
                "weighted_f1": baseline_metrics["weighted_f1"],
                "params": baseline_params,
                "notes": f"Flattened X shape {X.shape[1] * X.shape[2]}; split={split_info}",
            },
            {
                "model": "Strong Temporal SNN",
                "split_mode": split_mode,
                "normalization_mode": normalization_mode,
                "SEED_SNN_MODE": "strong",
                "accuracy": snn_metrics["accuracy"],
                "macro_f1": snn_metrics["macro_f1"],
                "weighted_f1": snn_metrics["weighted_f1"],
                "params": best_entry,
                "notes": (
                    f"Best of {len(grid_results)} configs; "
                    f"input (batch, {X.shape[1]}, {X.shape[2]}); split={split_info}"
                ),
            },
        ]

        csv_path, json_path = export_strong_seed_results(grid_results, best_entry)
        print("\nStrong SEED SNN results saved:")
        print(" ", csv_path)
        print(" ", json_path)
        return results

    print("\n--- Training SEED Temporal SNN ---")
    print("SEED_SNN_MODE: simple")
    print("Input shape:", X_train.shape)
    print("Output classes: 3")
    _, _, snn_y_test, snn_y_pred, snn_acc, snn_macro_f1, snn_params = train_seed_snn_model(
        X_train,
        y_train,
        X_test,
        y_test,
        num_classes=3,
    )
    snn_metrics = evaluate_seed_predictions(
        snn_y_test, snn_y_pred, model_name="SEED Temporal SNN"
    )

    print("\n=== SEED Final Comparison ===")
    print(f"Baseline Accuracy: {baseline_acc:.4f} | Macro F1: {baseline_macro_f1:.4f}")
    print(f"SNN Accuracy: {snn_acc:.4f} | Macro F1: {snn_macro_f1:.4f}")

    deap_note = "DEAP best Temporal SNN reference: ~53.12% accuracy, 0.5103 Macro F1"
    print(f"\n{deap_note}")
    if snn_acc > 0.5312:
        print("SEED SNN accuracy exceeds DEAP baseline.")
    else:
        print("Compare SEED vs DEAP using the metrics above.")

    results = [
        {
            "model": "Logistic Regression",
            "split_mode": split_mode,
            "normalization_mode": normalization_mode,
            "accuracy": baseline_metrics["accuracy"],
            "macro_f1": baseline_metrics["macro_f1"],
            "weighted_f1": baseline_metrics["weighted_f1"],
            "params": baseline_params,
            "notes": f"Flattened X shape {X.shape[1] * X.shape[2]}; split={split_info}",
        },
        {
            "model": "Temporal SNN",
            "split_mode": split_mode,
            "normalization_mode": normalization_mode,
            "SEED_SNN_MODE": "simple",
            "accuracy": snn_metrics["accuracy"],
            "macro_f1": snn_metrics["macro_f1"],
            "weighted_f1": snn_metrics["weighted_f1"],
            "params": snn_params,
            "notes": (
                f"Input (batch, {X.shape[1]}, {X.shape[2]}); "
                f"split={split_info}"
            ),
        },
    ]

    csv_path, json_path = export_seed_results(results)
    print("\nSEED results saved:")
    print(" ", csv_path)
    print(" ", json_path)

    return results
