from __future__ import annotations

from typing import Tuple

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


def train_baseline_model(
    X: np.ndarray,
    y: np.ndarray,
) -> Tuple[LogisticRegression, np.ndarray, np.ndarray, np.ndarray, float]:
    """
    Train a simple Logistic Regression baseline classifier.

    Returns:
        model, X_test, y_test, y_pred, accuracy
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    model = LogisticRegression(
        class_weight="balanced",
        random_state=42,
        max_iter=1000,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = float(accuracy_score(y_test, y_pred))

    return model, X_test, y_test, y_pred, acc

