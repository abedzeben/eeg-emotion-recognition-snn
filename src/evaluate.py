from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

from src.labels import BINARY_LABELS, EMOTION_LABELS


def _resolve_labels(y_true: np.ndarray, num_classes: int | None) -> tuple[list[int], list[str]]:
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


def evaluate_classification(
    y_true,
    y_pred,
    model_name: str,
    *,
    num_classes: int | None = None,
) -> None:
    """
    Print accuracy, confusion matrix, classification report, and macro F1.

    Supports binary (Calm/Excited) and 4-class Valence-Arousal emotions.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

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
