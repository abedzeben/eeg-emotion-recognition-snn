from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional, Tuple

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.archive.deap_research.deap_cnn_snn import DEAP_TEMPORAL_BASELINE, train_deap_temporal_baseline
from src.features import TEMPORAL_NUM_WINDOWS, extract_temporal_window_de_features
from src.labels import create_multi_emotion_labels

TEMPORAL_NORM_MODES = ("global", "per_subject", "per_subject_per_channel")
TemporalNormMode = Literal["global", "per_subject", "per_subject_per_channel"]

_SPLIT_RANDOM_STATE = 42
_EPS = 1e-8
NUM_BANDS = 5
NUM_DEAP_CHANNELS = 40


def normalize_deap_temporal_features(
    X_train: np.ndarray,
    X_test: np.ndarray,
    subjects_train: np.ndarray,
    subjects_test: np.ndarray,
    mode: TemporalNormMode,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Normalize (trials, windows, features) without changing representation shape.

    per_subject_per_channel uses internal channel grouping (40 ch × 5 bands)
    but returns the same (trials, windows, 200) layout.
    """
    if mode not in TEMPORAL_NORM_MODES:
        valid = ", ".join(TEMPORAL_NORM_MODES)
        raise ValueError(f"Unknown normalization mode '{mode}'. Valid: {valid}")

    if mode == "global":
        scaler = StandardScaler()
        n_features = X_train.shape[2]
        scaler.fit(X_train.reshape(-1, n_features))
        X_train_n = scaler.transform(X_train.reshape(-1, n_features)).reshape(X_train.shape)
        X_test_n = scaler.transform(X_test.reshape(-1, n_features)).reshape(X_test.shape)
        return X_train_n.astype(np.float32), X_test_n.astype(np.float32)

    if mode == "per_subject":
        out_train = np.empty_like(X_train, dtype=np.float32)
        out_test = np.empty_like(X_test, dtype=np.float32)
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
        n_trials_tr, n_windows, n_flat = X_train.shape
        n_channels = n_flat // NUM_BANDS
        if n_channels * NUM_BANDS != n_flat:
            raise ValueError(f"Expected {n_flat} features = channels × {NUM_BANDS}")

        X_train_ch = X_train.reshape(n_trials_tr, n_windows, n_channels, NUM_BANDS)
        n_trials_te = X_test.shape[0]
        X_test_ch = X_test.reshape(n_trials_te, n_windows, n_channels, NUM_BANDS)

        out_train = np.empty_like(X_train_ch, dtype=np.float32)
        out_test = np.empty_like(X_test_ch, dtype=np.float32)
        stats: Dict[tuple, tuple] = {}

        for subj in np.unique(subjects_train):
            mask = subjects_train == subj
            for ch in range(n_channels):
                block = X_train_ch[mask, :, ch, :]
                mean = float(np.mean(block))
                std = float(np.std(block)) + _EPS
                stats[(int(subj), ch)] = (mean, std)
                out_train[mask, :, ch, :] = (block - mean) / std

        global_mean = float(np.mean(X_train_ch))
        global_std = float(np.std(X_train_ch)) + _EPS

        for subj in np.unique(subjects_test):
            mask = subjects_test == subj
            for ch in range(n_channels):
                block = X_test_ch[mask, :, ch, :]
                mean, std = stats.get((int(subj), ch), (global_mean, global_std))
                out_test[mask, :, ch, :] = (block - mean) / std

        return (
            out_train.reshape(n_trials_tr, n_windows, n_flat).astype(np.float32),
            out_test.reshape(n_trials_te, n_windows, n_flat).astype(np.float32),
        )

    raise ValueError(f"Unhandled normalization mode: {mode}")


def print_temporal_norm_result(
    y_test: np.ndarray,
    y_pred: np.ndarray,
    *,
    normalization_mode: str,
    num_classes: int = 4,
) -> Dict[str, float]:
    """Print metrics for one normalization experiment."""
    acc = float(accuracy_score(y_test, y_pred))
    macro_f1 = float(f1_score(y_test, y_pred, average="macro", zero_division=0))

    print(f"\n--- Normalization: {normalization_mode} ---")
    print("Normalization mode:", normalization_mode)
    print("Accuracy:", f"{acc:.4f}")
    print("Macro F1:", f"{macro_f1:.4f}")
    print("Confusion matrix:")
    print(confusion_matrix(y_test, y_pred, labels=list(range(num_classes))))

    return {"normalization": normalization_mode, "accuracy": acc, "macro_f1": macro_f1}


def run_deap_temporal_normalization_study(
    *,
    folder: str = "data/raw",
    max_subjects: Optional[int] = None,
    label_strategy: str = "mean",
    trials_per_subject: int = 40,
) -> List[Dict[str, Any]]:
    """
    Step 40: Temporal SNN with subject-aware normalization (no CNN).
    """
    from src.load_data import load_all_deap_files
    from src.preprocessing import bandpass_filter

    print("\n" + "=" * 60)
    print("DEAP Temporal Normalization Study (Step 40)")
    print("=" * 60)
    print("Model: Temporal SNN (best fixed config)")
    print("Feature shape: (trials, windows, 200) — 40 channels × 5 DE bands")
    print("Label strategy:", label_strategy)
    print("Normalization modes:", list(TEMPORAL_NORM_MODES))
    print("Baseline reference:", DEAP_TEMPORAL_BASELINE)

    X, y_ratings = load_all_deap_files(folder, max_subjects=max_subjects)
    X_filtered = bandpass_filter(X)
    subject_ids = np.repeat(
        np.arange(X_filtered.shape[0] // trials_per_subject),
        trials_per_subject,
    )

    X_temporal = extract_temporal_window_de_features(
        X_filtered,
        num_windows=TEMPORAL_NUM_WINDOWS,
    )
    print("\nTemporal SNN feature shape:", X_temporal.shape)

    y_multi = create_multi_emotion_labels(y_ratings, strategy=label_strategy, verbose=False)

    (
        X_train,
        X_test,
        y_train,
        y_test,
        subj_train,
        subj_test,
    ) = train_test_split(
        X_temporal,
        y_multi,
        subject_ids,
        test_size=0.2,
        random_state=_SPLIT_RANDOM_STATE,
        stratify=y_multi,
    )
    print(f"Train samples: {X_train.shape[0]}, Test samples: {X_test.shape[0]}")

    results: List[Dict[str, Any]] = []

    for norm_mode in TEMPORAL_NORM_MODES:
        print(f"\n{'=' * 60}")
        print(f"Training Temporal SNN with normalization: {norm_mode}")

        X_train_n, X_test_n = normalize_deap_temporal_features(
            X_train,
            X_test,
            subj_train,
            subj_test,
            norm_mode,
        )

        y_pred, _ = train_deap_temporal_baseline(
            X_train_n,
            y_train,
            X_test_n,
            y_test,
            num_classes=4,
            apply_standard_scaler=False,
        )

        metrics = print_temporal_norm_result(
            y_test,
            y_pred,
            normalization_mode=norm_mode,
        )
        results.append(metrics)

    sorted_results = sorted(results, key=lambda r: r["macro_f1"], reverse=True)
    best = sorted_results[0]

    print("\n=== DEAP Temporal Normalization Summary ===")
    print("normalization | accuracy | macro_f1")
    for entry in sorted_results:
        print(
            f"{entry['normalization']} | {entry['accuracy']:.4f} | {entry['macro_f1']:.4f}"
        )

    print("\nBest normalization:", best["normalization"])
    print(f"Best Accuracy: {best['accuracy']:.4f}")
    print(f"Best Macro F1: {best['macro_f1']:.4f}")

    acc_delta = best["accuracy"] - DEAP_TEMPORAL_BASELINE["accuracy"]
    f1_delta = best["macro_f1"] - DEAP_TEMPORAL_BASELINE["macro_f1"]
    print("\nVs historical DEAP Temporal SNN baseline (~53.12% / 0.5103):")
    print(f"Accuracy delta: {acc_delta:+.4f}")
    print(f"Macro F1 delta: {f1_delta:+.4f}")

    return results
