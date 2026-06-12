from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Literal, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    recall_score,
)
from sklearn.preprocessing import StandardScaler

from src.seed_cnn_snn import (
    BEST_CNN_SNN_CONFIG,
    SEED_CNN_SNN_BASELINE,
    SEED_LR_BASELINE,
    train_cnn_snn_fixed_config,
)
from src.seed_experiment import (
    SEED_LABEL_NAMES,
    SEED_TEST_SUBJECTS,
    SEED_TRAIN_SUBJECTS,
    load_seed_dataset,
    print_seed_dataset_summary,
    split_seed_data,
)

SUBJECT_SHIFT_MODES = (
    "none",
    "global",
    "per_subject",
    "per_subject_per_channel",
    "per_band",
    "per_subject_per_band",
)
SubjectShiftMode = Literal[
    "none",
    "global",
    "per_subject",
    "per_subject_per_channel",
    "per_band",
    "per_subject_per_band",
]

_EPS = 1e-8

SUBJECT_SHIFT_FAST_MODES = (
    "global",
    "per_subject_per_channel",
    "per_subject_per_band",
)
SUBJECT_SHIFT_FAST_EPOCHS = 30


def _align_subject_splits(
    subjects: np.ndarray,
    X_train: np.ndarray,
    X_test: np.ndarray,
    split_mode: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """Subject IDs aligned with train/test tensors after split_seed_data."""
    if split_mode == "subject":
        return (
            subjects[np.isin(subjects, SEED_TRAIN_SUBJECTS)],
            subjects[np.isin(subjects, SEED_TEST_SUBJECTS)],
        )
    raise ValueError(
        "Subject shift study requires SEED_SPLIT_MODE='subject' "
        f"(got {split_mode!r})"
    )


def normalize_seed_subject_shift(
    X_train: np.ndarray,
    X_test: np.ndarray,
    subjects_train: np.ndarray,
    subjects_test: np.ndarray,
    mode: SubjectShiftMode,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Step 37 normalization strategies for SEED (samples, bands, channels).

    global: StandardScaler fit on training data only (per-channel features).
    Subject-aware modes use statistics from each subject's own samples.
    """
    if mode not in SUBJECT_SHIFT_MODES:
        valid = ", ".join(SUBJECT_SHIFT_MODES)
        raise ValueError(f"Unknown subject shift mode '{mode}'. Valid: {valid}")

    n_bands = X_train.shape[1]
    n_channels = X_train.shape[2]

    if mode == "none":
        return X_train.astype(np.float32, copy=True), X_test.astype(np.float32, copy=True)

    if mode == "global":
        scaler = StandardScaler()
        scaler.fit(X_train.reshape(-1, n_channels))
        X_train_n = scaler.transform(X_train.reshape(-1, n_channels)).reshape(X_train.shape)
        X_test_n = scaler.transform(X_test.reshape(-1, n_channels)).reshape(X_test.shape)
        return X_train_n.astype(np.float32), X_test_n.astype(np.float32)

    out_train = np.empty_like(X_train, dtype=np.float32)
    out_test = np.empty_like(X_test, dtype=np.float32)

    if mode == "per_subject":
        for subj in np.unique(subjects_train):
            mask = subjects_train == subj
            block = X_train[mask]
            mean = float(np.mean(block))
            std = float(np.std(block)) + _EPS
            out_train[mask] = (block - mean) / std
        for subj in np.unique(subjects_test):
            mask = subjects_test == subj
            block = X_test[mask]
            mean = float(np.mean(block))
            std = float(np.std(block)) + _EPS
            out_test[mask] = (block - mean) / std
        return out_train, out_test

    if mode == "per_subject_per_channel":
        for subj in np.unique(subjects_train):
            mask = subjects_train == subj
            for ch in range(n_channels):
                block = X_train[mask, :, ch]
                mean = float(np.mean(block))
                std = float(np.std(block)) + _EPS
                out_train[mask, :, ch] = (block - mean) / std
        for subj in np.unique(subjects_test):
            mask = subjects_test == subj
            for ch in range(n_channels):
                block = X_test[mask, :, ch]
                mean = float(np.mean(block))
                std = float(np.std(block)) + _EPS
                out_test[mask, :, ch] = (block - mean) / std
        return out_train, out_test

    if mode == "per_band":
        for band in range(n_bands):
            scaler = StandardScaler()
            scaler.fit(X_train[:, band, :])
            out_train[:, band, :] = scaler.transform(X_train[:, band, :])
            out_test[:, band, :] = scaler.transform(X_test[:, band, :])
        return out_train.astype(np.float32), out_test.astype(np.float32)

    if mode == "per_subject_per_band":
        for subj in np.unique(subjects_train):
            mask = subjects_train == subj
            for band in range(n_bands):
                block = X_train[mask, band, :]
                mean = float(np.mean(block))
                std = float(np.std(block)) + _EPS
                out_train[mask, band, :] = (block - mean) / std
        for subj in np.unique(subjects_test):
            mask = subjects_test == subj
            for band in range(n_bands):
                block = X_test[mask, band, :]
                mean = float(np.mean(block))
                std = float(np.std(block)) + _EPS
                out_test[mask, band, :] = (block - mean) / std
        return out_train, out_test

    raise ValueError(f"Unhandled subject shift mode: {mode}")


def evaluate_subject_shift_run(
    y_test: np.ndarray,
    y_pred: np.ndarray,
    *,
    normalization_mode: str,
    num_classes: int = 3,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Print and return metrics for one normalization experiment."""
    label_ids = list(range(num_classes))
    target_names = [SEED_LABEL_NAMES.get(i, f"Class {i}") for i in label_ids]

    acc = float(accuracy_score(y_test, y_pred))
    macro_f1 = float(f1_score(y_test, y_pred, average="macro", labels=label_ids, zero_division=0))
    weighted_f1 = float(
        f1_score(y_test, y_pred, average="weighted", labels=label_ids, zero_division=0)
    )
    recalls = recall_score(y_test, y_pred, average=None, labels=label_ids, zero_division=0)
    per_class_recall = {
        target_names[i]: float(recalls[i]) for i in range(len(label_ids))
    }

    print(f"\n--- Normalization: {normalization_mode} ---")
    print("Normalization mode:", normalization_mode)
    print("Accuracy:", f"{acc:.4f}")
    print("Macro F1:", f"{macro_f1:.4f}")
    print("Per-class recall:")
    for name, recall in per_class_recall.items():
        print(f"  {name}: {recall:.4f}")

    if verbose:
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
        "normalization": normalization_mode,
        "accuracy": acc,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "per_class_recall": per_class_recall,
    }


def print_subject_shift_study_summary(results: List[Dict[str, Any]]) -> None:
    """Print summary table sorted by Macro F1."""
    sorted_results = sorted(results, key=lambda r: r["macro_f1"], reverse=True)
    best = sorted_results[0]

    print("\n=== Subject Shift Study Summary ===")
    print("normalization | accuracy | macro_f1")
    for entry in sorted_results:
        print(
            f"{entry['normalization']} | {entry['accuracy']:.4f} | {entry['macro_f1']:.4f}"
        )

    print("\nBest normalization:", best["normalization"])
    print(f"Best Accuracy: {best['accuracy']:.4f}")
    print(f"Best Macro F1: {best['macro_f1']:.4f}")

    acc_delta = best["accuracy"] - SEED_CNN_SNN_BASELINE["accuracy"]
    f1_delta = best["macro_f1"] - SEED_CNN_SNN_BASELINE["macro_f1"]

    print("\nCompare against:")
    print(
        f"Logistic Regression: "
        f"{SEED_LR_BASELINE['accuracy']:.2%} Accuracy / "
        f"{SEED_LR_BASELINE['macro_f1']:.4f} Macro F1"
    )
    print(
        f"Previous CNN-SNN: "
        f"{SEED_CNN_SNN_BASELINE['accuracy']:.2%} Accuracy / "
        f"{SEED_CNN_SNN_BASELINE['macro_f1']:.4f} Macro F1"
    )
    print("\nImprovement over previous CNN-SNN:")
    print(f"Accuracy delta: {acc_delta:+.4f}")
    print(f"Macro F1 delta: {f1_delta:+.4f}")

    if best["macro_f1"] >= SEED_LR_BASELINE["macro_f1"]:
        print("Best normalization closes the gap to Logistic Regression on Macro F1.")
    elif f1_delta > 0:
        print("Subject-specific normalization improved CNN-SNN over the previous run.")
    else:
        print("No normalization mode exceeded the previous CNN-SNN Macro F1.")


def export_subject_shift_study(
    results: List[Dict[str, Any]],
    best_entry: Dict[str, Any],
    *,
    output_dir: Union[str, Path] = "results/metrics",
) -> Tuple[Path, Path]:
    """Export subject shift study results."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / "seed_subject_shift_study.csv"
    json_path = output_dir / "seed_subject_shift_study.json"

    rows = []
    for entry in results:
        rows.append(
            {
                "normalization": entry["normalization"],
                "accuracy": float(entry["accuracy"]),
                "macro_f1": float(entry["macro_f1"]),
                "weighted_f1": float(entry["weighted_f1"]),
                "best_epoch": entry.get("best_epoch"),
                "cnn_snn_config": json.dumps(entry.get("cnn_snn_config", BEST_CNN_SNN_CONFIG)),
                "is_best": entry["normalization"] == best_entry["normalization"],
            }
        )

    pd.DataFrame(rows).to_csv(csv_path, index=False)

    payload = {
        "study": "Step 37 subject shift normalization",
        "SEED_SNN_MODE": "cnn_snn",
        "fast_mode": results[0].get("fast_mode", False) if results else False,
        "fixed_cnn_snn_config": results[0].get("cnn_snn_config", BEST_CNN_SNN_CONFIG)
        if results
        else BEST_CNN_SNN_CONFIG,
        "baselines": {
            "logistic_regression": SEED_LR_BASELINE,
            "previous_cnn_snn": SEED_CNN_SNN_BASELINE,
        },
        "best": best_entry,
        "all_runs": results,
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=str)

    return csv_path, json_path


def run_seed_subject_shift_study(
    *,
    data_dir: Union[str, Path] = "data/seed",
    split_mode: str = "subject",
    cnn_snn_num_steps: int = 10,
    fast_mode: bool = False,
) -> List[Dict[str, Any]]:
    """
    Step 37: compare normalization strategies with fixed best CNN-SNN config.
    """
    if split_mode != "subject":
        raise ValueError("Subject shift study requires split_mode='subject'")

    norm_modes = SUBJECT_SHIFT_FAST_MODES if fast_mode else SUBJECT_SHIFT_MODES
    cnn_config = dict(BEST_CNN_SNN_CONFIG)
    if fast_mode:
        cnn_config["epochs"] = SUBJECT_SHIFT_FAST_EPOCHS

    print("\n" + "=" * 60)
    print("SEED Subject Shift Normalization Study (Step 37)")
    print("=" * 60)
    print("SEED_SNN_MODE: cnn_snn")
    print("SEED_SPLIT_MODE:", split_mode)
    print("SEED_SUBJECT_SHIFT_FAST:", fast_mode)
    print("Fixed CNN-SNN config (no hyperparameter search):", cnn_config)
    print("CNN_SNN_NUM_STEPS:", cnn_snn_num_steps)
    print("Normalization modes:", list(norm_modes))
    if fast_mode:
        print("Fast mode: compact metrics only (no full classification report)")

    X, y, subjects = load_seed_dataset(data_dir)
    print_seed_dataset_summary(X, y, subjects)

    X_train, X_test, y_train, y_test, split_info = split_seed_data(
        X, y, subjects, split_mode="subject"
    )
    subjects_train, subjects_test = _align_subject_splits(
        subjects, X_train, X_test, split_mode="subject"
    )
    print(f"\nTrain samples: {X_train.shape[0]}, Test samples: {X_test.shape[0]}")
    print("Input shape:", X_train.shape)

    results: List[Dict[str, Any]] = []

    for norm_mode in norm_modes:
        print(f"\n{'=' * 60}")
        print(f"Testing normalization: {norm_mode}")

        X_train_n, X_test_n = normalize_seed_subject_shift(
            X_train,
            X_test,
            subjects_train,
            subjects_test,
            norm_mode,
        )

        y_pred, train_info = train_cnn_snn_fixed_config(
            X_train_n,
            y_train,
            X_test_n,
            y_test,
            num_steps=cnn_snn_num_steps,
            config=cnn_config,
        )

        metrics = evaluate_subject_shift_run(
            y_test,
            y_pred,
            normalization_mode=norm_mode,
            verbose=not fast_mode,
        )
        entry = {
            **metrics,
            "best_epoch": train_info.get("best_epoch"),
            "cnn_snn_config": dict(cnn_config),
            "num_steps": cnn_snn_num_steps,
            "split_info": split_info,
            "fast_mode": fast_mode,
        }
        results.append(entry)

    sorted_results = sorted(results, key=lambda r: r["macro_f1"], reverse=True)
    best_entry = sorted_results[0]
    print_subject_shift_study_summary(results)

    csv_path, json_path = export_subject_shift_study(results, best_entry)
    print("\nSubject shift study results saved:")
    print(" ", csv_path)
    print(" ", json_path)

    return results
