from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


def train_random_forest_model(
    X: np.ndarray,
    y: np.ndarray,
) -> Tuple[Pipeline, np.ndarray, np.ndarray, np.ndarray, float, float, Dict[str, Any]]:
    """
    Train and tune a Random Forest with SelectKBest feature selection.

    Grid search over k, n_estimators, max_depth, and class_weight.
    Best configuration selected by macro F1.

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

    k_values = [50, 100, 200, 300]
    n_estimators_options = [100, 200]
    max_depth_options: list = [None, 10, 20]
    class_weight_options: list = [None, "balanced"]

    n_features = X_train.shape[1]
    valid_k_values = [k for k in k_values if k <= n_features]

    best_model: Pipeline | None = None
    best_pred: np.ndarray | None = None
    best_params: Dict[str, Any] = {}
    best_macro_f1 = -1.0
    best_acc = 0.0

    for k in valid_k_values:
        for n_est in n_estimators_options:
            for max_depth in max_depth_options:
                for cw in class_weight_options:
                    pipe = Pipeline(
                        [
                            ("select", SelectKBest(score_func=f_classif, k=k)),
                            (
                                "clf",
                                RandomForestClassifier(
                                    n_estimators=n_est,
                                    max_depth=max_depth,
                                    class_weight=cw,
                                    random_state=42,
                                    n_jobs=-1,
                                ),
                            ),
                        ]
                    )
                    pipe.fit(X_train, y_train)
                    y_pred = pipe.predict(X_test)
                    acc = float(accuracy_score(y_test, y_pred))
                    macro_f1 = float(
                        f1_score(y_test, y_pred, average="macro", zero_division=0)
                    )

                    print(
                        f"RF config k={k}, n_estimators={n_est}, max_depth={max_depth}, "
                        f"class_weight={cw} | acc={acc:.4f} macro_f1={macro_f1:.4f}"
                    )

                    if macro_f1 > best_macro_f1:
                        best_macro_f1 = macro_f1
                        best_acc = acc
                        best_model = pipe
                        best_pred = y_pred
                        best_params = {
                            "k": k,
                            "n_estimators": n_est,
                            "max_depth": max_depth,
                            "class_weight": cw,
                        }

    assert best_model is not None and best_pred is not None
    print(f"Selected Random Forest params: {best_params} | macro F1: {best_macro_f1:.4f}")

    return best_model, X_test, y_test, best_pred, best_acc, best_macro_f1, best_params
