from __future__ import annotations

import numpy as np
from typing import List, Optional, Tuple
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    recall_score,
)

from src.labels import BINARY_LABELS, EMOTION_LABELS


def _to_class_labels(y: np.ndarray) -> np.ndarray:
    """Ensure 1D integer class labels for sklearn metrics (not probability vectors)."""
    arr = np.asarray(y)
    if arr.ndim == 2:
        arr = np.argmax(arr, axis=1)
    return arr.ravel().astype(int)


def _resolve_labels(y_true: np.ndarray, num_classes: Optional[int]) -> Tuple[List[int], List[str]]:
    if num_classes is None:
        num_classes = int(max(y_true.max(), 0)) + 1
        unique = np.unique(y_true)
        if len(unique) > num_classes:
            num_classes = int(unique.max()) + 1

    if num_classes <= 2:
        label_ids = [0, 1]
        target_names = [BINARY_LABELS[i] for i in label_ids]
    else:
        label_ids = list(range(num_classes))
        target_names = [EMOTION_LABELS.get(i, f"Class {i}") for i in label_ids]

    return label_ids, target_names


def evaluate_snn_research_experiment(
    y_true,
    y_pred,
    *,
    normalization_mode: str,
    feature_type: str,
    feature_shape: tuple,
    num_classes: int = 4,
) -> dict:
    """
    Step 33: print metrics for one SNN research experiment and return summary dict.
    """
    y_true = _to_class_labels(y_true)
    y_pred = _to_class_labels(y_pred)
    label_ids, target_names = _resolve_labels(y_true, num_classes)

    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro", labels=label_ids, zero_division=0)
    recalls = recall_score(y_true, y_pred, average=None, labels=label_ids, zero_division=0)

    print("\n--- SNN research experiment ---")
    print("Normalization mode:", normalization_mode)
    print("Feature type:", feature_type)
    print("Temporal feature shape:", feature_shape)
    print("Accuracy:", f"{acc:.4f}")
    print("Macro F1:", f"{macro_f1:.4f}")
    print("Confusion matrix:")
    print(confusion_matrix(y_true, y_pred, labels=label_ids))
    print("Recall per class:")
    for label_id, name, recall in zip(label_ids, target_names, recalls):
        print(f"  {name}: {recall:.4f}")

    return {
        "normalization": normalization_mode,
        "feature_type": feature_type,
        "feature_shape": tuple(feature_shape),
        "accuracy": acc,
        "macro_f1": macro_f1,
        "recalls": {target_names[i]: float(recalls[i]) for i in range(len(label_ids))},
    }


def print_snn_research_summary(results: list) -> None:
    """Print Step 33 research summary sorted by macro F1."""
    if not results:
        print("\n=== SNN Research Summary ===")
        print("No experiments completed.")
        return

    sorted_results = sorted(results, key=lambda r: r["macro_f1"], reverse=True)
    best = sorted_results[0]

    print("\n=== SNN Research Summary ===")
    print("Sorted by Macro F1:")
    print("normalization | feature_type | accuracy | macro_f1")
    for entry in sorted_results:
        print(
            f"{entry['normalization']} | {entry['feature_type']} | "
            f"{entry['accuracy']:.4f} | {entry['macro_f1']:.4f}"
        )

    print("\nBest configuration:")
    print("Normalization mode:", best["normalization"])
    print("Feature type:", best["feature_type"])
    print("Temporal feature shape:", best["feature_shape"])
    print("Accuracy:", f"{best['accuracy']:.4f}")
    print("Macro F1:", f"{best['macro_f1']:.4f}")


def evaluate_classification(
    y_true,
    y_pred,
    model_name: str,
    *,
    num_classes: Optional[int] = None,
) -> None:
    """
    Print accuracy, confusion matrix, classification report, and macro F1.

    Supports binary (Calm/Excited) and 4-class Valence-Arousal emotions.
    """
    y_true = _to_class_labels(y_true)
    y_pred = _to_class_labels(y_pred)

    label_ids, target_names = _resolve_labels(y_true, num_classes)

    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro", labels=label_ids, zero_division=0)

    print(f"\n=== {model_name} ===")
    print("Accuracy:", acc)
    print("Macro F1:", macro_f1)
    print("Confusion matrix:")
    print(confusion_matrix(y_true, y_pred, labels=label_ids))
    print("Classification report:")
    print(
        classification_report(
            y_true,
            y_pred,
            labels=label_ids,
            target_names=target_names,
            zero_division=0,
        )
    )
