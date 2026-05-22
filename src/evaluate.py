from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


def evaluate_classification(y_true, y_pred, model_name: str) -> None:
    """
    Print accuracy, confusion matrix, and classification report for binary arousal labels.

    Labels:
      0 = Calm
      1 = Excited
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    target_names = ["Calm", "Excited"]

    acc = accuracy_score(y_true, y_pred)

    print(f"\n=== {model_name} ===")
    print("Accuracy:", acc)
    print("Confusion matrix:")
    print(confusion_matrix(y_true, y_pred, labels=[0, 1]))
    print("Classification report:")
    print(
        classification_report(
            y_true,
            y_pred,
            labels=[0, 1],
            target_names=target_names,
            zero_division=0,
        )
    )
