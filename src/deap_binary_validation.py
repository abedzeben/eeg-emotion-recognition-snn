from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

from src.deap_cnn_snn import DEAP_TEMPORAL_BASELINE
from src.features import TEMPORAL_NUM_WINDOWS, extract_temporal_window_de_features
from src.labels import (
    AROUSAL_BINARY_LABELS,
    BINARY_VALIDATION_THRESHOLD,
    VALENCE_BINARY_LABELS,
    create_arousal_binary_labels,
    create_valence_binary_labels,
    print_binary_class_distribution,
)
from src.snn_model import BEST_TEMPORAL_SNN_CONFIG, train_tuned_snn_model

_SPLIT_RANDOM_STATE = 42
_TEST_SIZE = 0.2
METRICS_PATH = Path("results/metrics/deap_binary_validation.json")


def _evaluate_binary_experiment(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    label_map: Dict[int, str],
    *,
    model_name: str,
) -> Dict[str, Any]:
    label_ids = [0, 1]
    target_names = [label_map[i] for i in label_ids]

    acc = float(accuracy_score(y_true, y_pred))
    macro_f1 = float(
        f1_score(y_true, y_pred, average="macro", labels=label_ids, zero_division=0)
    )
    cm = confusion_matrix(y_true, y_pred, labels=label_ids)
    report_str = classification_report(
        y_true,
        y_pred,
        labels=label_ids,
        target_names=target_names,
        zero_division=0,
    )
    report_dict = classification_report(
        y_true,
        y_pred,
        labels=label_ids,
        target_names=target_names,
        zero_division=0,
        output_dict=True,
    )

    print(f"\n=== {model_name} ===")
    print("Accuracy:", f"{acc:.4f}")
    print("Macro F1:", f"{macro_f1:.4f}")
    print("Confusion matrix:")
    print(cm)
    print("Classification report:")
    print(report_str)

    return {
        "accuracy": acc,
        "macro_f1": macro_f1,
        "confusion_matrix": cm.tolist(),
        "classification_report": report_dict,
        "classification_report_text": report_str,
    }


def _run_single_binary_experiment(
    X_temporal: np.ndarray,
    y_binary: np.ndarray,
    *,
    task_name: str,
    label_map: Dict[int, str],
) -> Dict[str, Any]:
    print(f"\n{'=' * 60}")
    print(task_name)
    print("=" * 60)
    print("Threshold:", BINARY_VALIDATION_THRESHOLD)
    print("Model: Temporal SNN (Step 29 best config)")
    print("Config:", BEST_TEMPORAL_SNN_CONFIG)
    print(
        "Features: windowed DE,",
        f"{TEMPORAL_NUM_WINDOWS} windows × 200 features",
    )

    class_distribution = print_binary_class_distribution(
        y_binary,
        label_map,
        title="Class distribution (full dataset)",
    )

    n_total = X_temporal.shape[0]
    n_test = int(round(n_total * _TEST_SIZE))
    n_train = n_total - n_test
    print(f"\nTrain samples: {n_train}, Test samples: {n_test}")
    print(
        f"Split: test_size={_TEST_SIZE}, random_state={_SPLIT_RANDOM_STATE}, stratified"
    )

    (
        _model,
        _X_test,
        y_test,
        y_pred,
        accuracy,
        macro_f1,
        params,
    ) = train_tuned_snn_model(
        X_temporal,
        y_binary,
        temporal=True,
        use_best_temporal_config=True,
        temporal_spike_encoding=False,
        quiet=False,
    )

    metrics = _evaluate_binary_experiment(
        y_test,
        y_pred,
        label_map,
        model_name=f"Temporal SNN — {task_name}",
    )

    return {
        "task": task_name,
        "model": "Temporal SNN (Step 29 best config)",
        "threshold": BINARY_VALIDATION_THRESHOLD,
        "label_map": label_map,
        "class_distribution_full": class_distribution,
        "train_samples": n_train,
        "test_samples": n_test,
        "split": {
            "test_size": _TEST_SIZE,
            "random_state": _SPLIT_RANDOM_STATE,
            "stratify": True,
        },
        "accuracy": metrics["accuracy"],
        "macro_f1": metrics["macro_f1"],
        "confusion_matrix": metrics["confusion_matrix"],
        "classification_report": metrics["classification_report"],
        "params": params,
    }


def run_deap_binary_validation(
    folder: str = "data/raw",
    *,
    max_subjects: Optional[int] = None,
    trials_per_subject: int = 40,
    output_path: Path = METRICS_PATH,
) -> Dict[str, Any]:
    """
    Step 43: binary Valence and Arousal Temporal SNN validation on full DEAP.
    """
    from src.load_data import load_all_deap_files
    from src.preprocessing import bandpass_filter

    print("\n" + "=" * 60)
    print("DEAP Binary Valence/Arousal Validation (Step 43)")
    print("=" * 60)
    print("Purpose: compare binary tasks vs 4-class Temporal SNN baseline")
    print("Reference 4-class baseline:", DEAP_TEMPORAL_BASELINE)
    print("Using full DEAP dataset" if max_subjects is None else f"max_subjects={max_subjects}")

    X, y_ratings = load_all_deap_files(folder, max_subjects=max_subjects)
    X_filtered = bandpass_filter(X)

    X_temporal = extract_temporal_window_de_features(
        X_filtered,
        num_windows=TEMPORAL_NUM_WINDOWS,
    )
    print("\nTemporal SNN feature shape:", X_temporal.shape)

    n_subjects = X_temporal.shape[0] // trials_per_subject
    dataset_info = {
        "subjects": n_subjects,
        "trials": int(X_temporal.shape[0]),
        "temporal_shape": list(X_temporal.shape),
        "trials_per_subject": trials_per_subject,
    }
    print("Dataset:", dataset_info)

    valence_labels = create_valence_binary_labels(y_ratings)
    arousal_labels = create_arousal_binary_labels(y_ratings)

    valence_result = _run_single_binary_experiment(
        X_temporal,
        valence_labels,
        task_name="Experiment 1 — Valence Binary Classification",
        label_map=VALENCE_BINARY_LABELS,
    )
    arousal_result = _run_single_binary_experiment(
        X_temporal,
        arousal_labels,
        task_name="Experiment 2 — Arousal Binary Classification",
        label_map=AROUSAL_BINARY_LABELS,
    )

    val_acc = valence_result["accuracy"]
    val_f1 = valence_result["macro_f1"]
    aro_acc = arousal_result["accuracy"]
    aro_f1 = arousal_result["macro_f1"]
    ref_acc = DEAP_TEMPORAL_BASELINE["accuracy"]
    ref_f1 = DEAP_TEMPORAL_BASELINE["macro_f1"]

    print("\n" + "=" * 60)
    print("=== Step 43 Comparison Summary ===")
    print("=" * 60)
    print(f"{'Task':<42} {'Accuracy':>10} {'Macro F1':>10}")
    print(f"{'4-class Temporal SNN (reference)':<42} {ref_acc:>10.4f} {ref_f1:>10.4f}")
    print(f"{'Binary Valence (this run)':<42} {val_acc:>10.4f} {val_f1:>10.4f}")
    print(f"{'Binary Arousal (this run)':<42} {aro_acc:>10.4f} {aro_f1:>10.4f}")
    print("\nDelta vs 4-class reference:")
    print(f"  Valence accuracy: {val_acc - ref_acc:+.4f}")
    print(f"  Valence Macro F1: {val_f1 - ref_f1:+.4f}")
    print(f"  Arousal accuracy: {aro_acc - ref_acc:+.4f}")
    print(f"  Arousal Macro F1: {aro_f1 - ref_f1:+.4f}")

    payload: Dict[str, Any] = {
        "study": "Step 43 DEAP Binary Valence/Arousal Validation",
        "threshold": BINARY_VALIDATION_THRESHOLD,
        "reference_4class_baseline": DEAP_TEMPORAL_BASELINE,
        "model_config": BEST_TEMPORAL_SNN_CONFIG,
        "features": {
            "type": "windowed_differential_entropy",
            "num_windows": TEMPORAL_NUM_WINDOWS,
            "features_per_window": 200,
            "channels": 40,
            "bands": 5,
        },
        "split": {
            "test_size": _TEST_SIZE,
            "random_state": _SPLIT_RANDOM_STATE,
            "stratify": True,
        },
        "dataset": dataset_info,
        "experiments": [valence_result, arousal_result],
        "comparison": {
            "valence_vs_4class": {
                "accuracy_delta": val_acc - ref_acc,
                "macro_f1_delta": val_f1 - ref_f1,
            },
            "arousal_vs_4class": {
                "accuracy_delta": aro_acc - ref_acc,
                "macro_f1_delta": aro_f1 - ref_f1,
            },
        },
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(f"\nStep 43 results saved: {output_path}")
    return payload
