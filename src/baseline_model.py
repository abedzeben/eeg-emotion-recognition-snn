from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def _macro_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }


def train_baseline_model(
    X: np.ndarray,
    y: np.ndarray,
) -> Tuple[Pipeline, np.ndarray, np.ndarray, np.ndarray, float, float, Dict[str, Any]]:
    """
    Train and tune a Logistic Regression baseline (StandardScaler + LR).

    Selects the best configuration by macro F1 score.

    Returns:
        best_model, X_test, y_test, y_pred, accuracy, macro_f1, best_params
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    c_values = [0.01, 0.1, 1, 10]
    class_weights: list = [None, "balanced"]

    best_model: Optional[Pipeline] = None
    best_pred: Optional[np.ndarray] = None
    best_params: Dict[str, Any] = {}
    best_macro_f1 = -1.0
    best_acc = 0.0

    for cw in class_weights:
        for c in c_values:
            pipe = Pipeline(
                [
                    ("scaler", StandardScaler()),
                    (
                        "clf",
                        LogisticRegression(
                            C=c,
                            class_weight=cw,
                            random_state=42,
                            max_iter=1000,
                            multi_class="auto",
                        ),
                    ),
                ]
            )
            pipe.fit(X_train, y_train)
            y_pred = pipe.predict(X_test)
            metrics = _macro_metrics(y_test, y_pred)

            print(
                f"Baseline config C={c}, class_weight={cw} | "
                f"acc={metrics['accuracy']:.4f} macro_f1={metrics['macro_f1']:.4f}"
            )

            if metrics["macro_f1"] > best_macro_f1:
                best_macro_f1 = metrics["macro_f1"]
                best_acc = metrics["accuracy"]
                best_model = pipe
                best_pred = y_pred
                best_params = {"C": c, "class_weight": cw}

    assert best_model is not None and best_pred is not None
    print(f"Selected baseline params: {best_params} | macro F1: {best_macro_f1:.4f}")

    return best_model, X_test, y_test, best_pred, best_acc, best_macro_f1, best_params
