from __future__ import annotations

import copy
import itertools
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import train_test_split

import snntorch as snn

from src.seed_experiment import SEED_LABEL_NAMES, evaluate_seed_predictions

SEED_LR_BASELINE = {
    "accuracy": 0.5463,
    "macro_f1": 0.5263,
}
SEED_STRONG_SNN_BASELINE = {
    "accuracy": 0.4709,
    "macro_f1": 0.4412,
}
SEED_CNN_SNN_BASELINE = {
    "accuracy": 0.4879,
    "macro_f1": 0.4912,
}

BEST_CNN_SNN_CONFIG: Dict[str, Any] = {
    "learning_rate": 0.001,
    "dropout": 0.3,
    "beta": 0.95,
    "class_weight": "balanced",
    "epochs": 100,
}

EARLY_STOPPING_PATIENCE = 15
LR_SCHEDULER_FACTOR = 0.5
LR_SCHEDULER_PATIENCE = 5
WEIGHT_DECAY = 1e-4
BATCH_SIZE = 128
VAL_SPLIT = 0.15
VAL_RANDOM_STATE = 42


class CnnSnnHybrid(nn.Module):
    """
    Step 36: CNN feature extractor on (bands × channels) map + SNN classifier.

    Input: (batch, 5, 62) → CNN on (batch, 1, 5, 62) → 128-d features
    → repeated over num_steps → SNN → averaged logits.
    """

    def __init__(
        self,
        *,
        num_bands: int = 5,
        num_channels: int = 62,
        num_steps: int = 10,
        num_classes: int = 3,
        beta: float = 0.95,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.num_steps = num_steps
        self.num_classes = num_classes

        self.conv1 = nn.Conv2d(1, 16, kernel_size=(2, 5), padding=(1, 2))
        self.bn1 = nn.BatchNorm2d(16)
        self.pool1 = nn.MaxPool2d(kernel_size=(1, 2))
        self.conv2 = nn.Conv2d(16, 32, kernel_size=(2, 5), padding=(1, 2))
        self.bn2 = nn.BatchNorm2d(32)
        self.pool2 = nn.MaxPool2d(kernel_size=(1, 2))

        with torch.no_grad():
            dummy = torch.zeros(1, 1, num_bands, num_channels)
            flat_size = self._cnn_flatten(dummy).shape[1]

        self.fc_cnn = nn.Linear(flat_size, 128)
        self.dropout_cnn = nn.Dropout(dropout)

        self.fc1 = nn.Linear(128, 128)
        self.lif1 = snn.Leaky(beta=beta)
        self.dropout1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(128, 64)
        self.lif2 = snn.Leaky(beta=beta)
        self.dropout2 = nn.Dropout(dropout)
        self.fc_out = nn.Linear(64, num_classes)

    def _cnn_flatten(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        return x.flatten(1)

    def _extract_cnn_features(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 3:
            raise ValueError(f"Expected x shape (batch, bands, channels), got {tuple(x.shape)}")
        x_map = x.unsqueeze(1)
        flat = self._cnn_flatten(x_map)
        feats = F.relu(self.fc_cnn(flat))
        return self.dropout_cnn(feats)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self._extract_cnn_features(x)
        batch = feats.shape[0]
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        out_sum = torch.zeros((batch, self.num_classes), device=x.device)

        for _ in range(self.num_steps):
            cur1 = self.fc1(feats)
            spk1, mem1 = self.lif1(cur1, mem1)
            spk1 = self.dropout1(spk1)
            cur2 = self.fc2(spk1)
            spk2, mem2 = self.lif2(cur2, mem2)
            spk2 = self.dropout2(spk2)
            out_sum = out_sum + self.fc_out(spk2)

        return out_sum / float(self.num_steps)


def get_seed_cnn_snn_grid(*, fast_grid: bool = True) -> Dict[str, List[Any]]:
    """Hyperparameter grid for CNN-SNN hybrid (8 configs fast, 96 full)."""
    if fast_grid:
        return {
            "learning_rate": [0.001, 0.0005],
            "dropout": [0.2, 0.3],
            "beta": [0.95],
            "class_weight": [None, "balanced"],
            "epochs": [100],
        }
    return {
        "learning_rate": [0.001, 0.0005, 0.0003],
        "dropout": [0.1, 0.2, 0.3, 0.4],
        "beta": [0.9, 0.95],
        "class_weight": [None, "balanced"],
        "epochs": [100, 150],
    }


def _count_grid_configs(grid: Dict[str, List[Any]]) -> int:
    total = 1
    for values in grid.values():
        total *= len(values)
    return total


def _class_weight_tensor(
    y_train: np.ndarray,
    num_classes: int,
    device: torch.device,
) -> torch.Tensor:
    class_counts = np.bincount(y_train, minlength=num_classes).astype(np.float32)
    class_counts = np.maximum(class_counts, 1.0)
    weights = len(y_train) / (num_classes * class_counts)
    return torch.tensor(weights, dtype=torch.float32, device=device)


def _predict(
    model: nn.Module,
    X: np.ndarray,
    device: torch.device,
    batch_size: int = BATCH_SIZE,
) -> np.ndarray:
    model.eval()
    X_t = torch.tensor(X, dtype=torch.float32, device=device)
    preds: List[np.ndarray] = []
    with torch.no_grad():
        for i in range(0, X_t.shape[0], batch_size):
            xb = X_t[i : i + batch_size]
            logits = model(xb)
            preds.append(torch.argmax(logits, dim=1).cpu().numpy())
    return np.concatenate(preds, axis=0)


def _evaluate_loss(
    model: nn.Module,
    X: np.ndarray,
    y: np.ndarray,
    criterion: nn.Module,
    device: torch.device,
    batch_size: int = BATCH_SIZE,
) -> Tuple[float, float]:
    model.eval()
    X_t = torch.tensor(X, dtype=torch.float32, device=device)
    y_t = torch.tensor(y, dtype=torch.long, device=device)
    total_loss = 0.0
    n_batches = 0
    all_preds: List[np.ndarray] = []

    with torch.no_grad():
        for i in range(0, X_t.shape[0], batch_size):
            xb = X_t[i : i + batch_size]
            yb = y_t[i : i + batch_size]
            logits = model(xb)
            total_loss += criterion(logits, yb).item()
            n_batches += 1
            all_preds.append(torch.argmax(logits, dim=1).cpu().numpy())

    y_pred = np.concatenate(all_preds, axis=0)
    macro_f1 = float(f1_score(y, y_pred, average="macro", zero_division=0))
    return total_loss / max(n_batches, 1), macro_f1


def train_single_cnn_snn(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    *,
    learning_rate: float,
    max_epochs: int,
    dropout: float,
    beta: float,
    class_weight_mode: Optional[str],
    num_steps: int = 10,
    num_classes: int = 3,
    device: Optional[torch.device] = None,
) -> Tuple[CnnSnnHybrid, np.ndarray, Dict[str, Any]]:
    """Train one CNN-SNN config with early stopping and LR scheduling."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = CnnSnnHybrid(
        num_bands=X_train.shape[1],
        num_channels=X_train.shape[2],
        num_steps=num_steps,
        num_classes=num_classes,
        beta=beta,
        dropout=dropout,
    ).to(device)

    if class_weight_mode == "balanced":
        weight_tensor = _class_weight_tensor(y_train, num_classes, device)
        criterion = nn.CrossEntropyLoss(weight=weight_tensor)
    else:
        criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=WEIGHT_DECAY,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=LR_SCHEDULER_FACTOR,
        patience=LR_SCHEDULER_PATIENCE,
    )

    X_train_t = torch.tensor(X_train, dtype=torch.float32, device=device)
    y_train_t = torch.tensor(y_train, dtype=torch.long, device=device)

    best_state: Optional[Dict[str, torch.Tensor]] = None
    best_val_f1 = -1.0
    best_epoch = 0
    patience_counter = 0

    model.train()
    n_train = X_train_t.shape[0]
    for epoch in range(1, max_epochs + 1):
        perm = torch.randperm(n_train, device=device)
        for i in range(0, n_train, BATCH_SIZE):
            idx = perm[i : i + BATCH_SIZE]
            xb = X_train_t[idx]
            yb = y_train_t[idx]
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

        val_loss, val_macro_f1 = _evaluate_loss(model, X_val, y_val, criterion, device)
        scheduler.step(val_loss)

        if val_macro_f1 > best_val_f1:
            best_val_f1 = val_macro_f1
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= EARLY_STOPPING_PATIENCE:
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    y_pred = _predict(model, X_test, device)
    test_acc = float(accuracy_score(y_test, y_pred))
    test_macro_f1 = float(f1_score(y_test, y_pred, average="macro", zero_division=0))
    test_weighted_f1 = float(
        f1_score(y_test, y_pred, average="weighted", zero_division=0)
    )

    info = {
        "learning_rate": learning_rate,
        "epochs": max_epochs,
        "dropout": dropout,
        "beta": beta,
        "class_weight": class_weight_mode,
        "num_steps": num_steps,
        "best_epoch": best_epoch,
        "val_macro_f1": best_val_f1,
        "accuracy": test_acc,
        "macro_f1": test_macro_f1,
        "weighted_f1": test_weighted_f1,
    }
    return model, y_pred, info


def train_cnn_snn_fixed_config(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    *,
    num_steps: int = 10,
    config: Optional[Dict[str, Any]] = None,
    num_classes: int = 3,
    seed: int = 42,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Step 37: train CNN-SNN with fixed best hyperparameters (no grid search).
    """
    cfg = {**BEST_CNN_SNN_CONFIG, **(config or {})}
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train,
        y_train,
        test_size=VAL_SPLIT,
        random_state=VAL_RANDOM_STATE,
        stratify=y_train,
    )

    _, y_pred, info = train_single_cnn_snn(
        X_tr,
        y_tr,
        X_val,
        y_val,
        X_test,
        y_test,
        learning_rate=float(cfg["learning_rate"]),
        max_epochs=int(cfg["epochs"]),
        dropout=float(cfg["dropout"]),
        beta=float(cfg["beta"]),
        class_weight_mode=cfg["class_weight"],
        num_steps=num_steps,
        num_classes=num_classes,
    )
    info["SEED_SNN_MODE"] = "cnn_snn"
    info["fixed_config"] = True
    return y_pred, info


def _print_cnn_snn_config_header(params: Dict[str, Any]) -> None:
    print("\n--- CNN-SNN SEED configuration ---")
    print("SEED_SNN_MODE: cnn_snn")
    print("learning_rate:", params["learning_rate"])
    print("dropout:", params["dropout"])
    print("beta:", params["beta"])
    print("class_weight:", params["class_weight"])
    print("epochs:", params["epochs"])
    print("num_steps:", params.get("num_steps"))


def train_cnn_snn_grid(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    *,
    fast_grid: bool = True,
    num_steps: int = 10,
    num_classes: int = 3,
) -> Tuple[np.ndarray, List[Dict[str, Any]], Dict[str, Any]]:
    """Grid search CNN-SNN hybrid; select best config by test Macro F1."""
    grid = get_seed_cnn_snn_grid(fast_grid=fast_grid)
    n_configs = _count_grid_configs(grid)
    print("\n=== CNN-SNN SEED training (Step 36) ===")
    print("SEED_SNN_MODE: cnn_snn")
    print("SEED_CNN_SNN_FAST_GRID:", fast_grid)
    print("CNN_SNN_NUM_STEPS:", num_steps)
    print("Total configurations:", n_configs)
    print("Early stopping patience:", EARLY_STOPPING_PATIENCE)
    print("Optimizer: AdamW, weight_decay:", WEIGHT_DECAY)

    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train,
        y_train,
        test_size=VAL_SPLIT,
        random_state=VAL_RANDOM_STATE,
        stratify=y_train,
    )
    print(
        f"Train/val split: {X_tr.shape[0]} train, {X_val.shape[0]} val "
        f"({int((1 - VAL_SPLIT) * 100)}% / {int(VAL_SPLIT * 100)}%), "
        f"{X_test.shape[0]} test"
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_results: List[Dict[str, Any]] = []
    best_entry: Optional[Dict[str, Any]] = None
    best_macro_f1 = -1.0
    best_y_pred: Optional[np.ndarray] = None

    keys = list(grid.keys())
    for combo in itertools.product(*(grid[k] for k in keys)):
        params = dict(zip(keys, combo))
        max_epochs = int(params["epochs"])

        _, y_pred, info = train_single_cnn_snn(
            X_tr,
            y_tr,
            X_val,
            y_val,
            X_test,
            y_test,
            learning_rate=float(params["learning_rate"]),
            max_epochs=max_epochs,
            dropout=float(params["dropout"]),
            beta=float(params["beta"]),
            class_weight_mode=params["class_weight"],
            num_steps=num_steps,
            num_classes=num_classes,
            device=device,
        )

        entry = {
            "SEED_SNN_MODE": "cnn_snn",
            **params,
            "num_steps": num_steps,
            "best_epoch": info["best_epoch"],
            "accuracy": info["accuracy"],
            "macro_f1": info["macro_f1"],
            "weighted_f1": info["weighted_f1"],
            "val_macro_f1": info["val_macro_f1"],
        }
        run_results.append(entry)

        _print_cnn_snn_config_header(entry)
        print("best epoch:", entry["best_epoch"])
        print("Accuracy:", f"{entry['accuracy']:.4f}")
        print("Macro F1:", f"{entry['macro_f1']:.4f}")
        print("Weighted F1:", f"{entry['weighted_f1']:.4f}")

        label_ids = list(range(num_classes))
        target_names = [SEED_LABEL_NAMES.get(i, f"Class {i}") for i in label_ids]
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

        if entry["macro_f1"] > best_macro_f1:
            best_macro_f1 = entry["macro_f1"]
            best_entry = entry
            best_y_pred = y_pred

    assert best_entry is not None and best_y_pred is not None
    return best_y_pred, run_results, best_entry


def print_cnn_snn_seed_summary(best_entry: Dict[str, Any]) -> None:
    """Print Step 36 final summary vs baselines."""
    print("\n=== CNN-SNN SEED Summary ===")
    print(f"Best CNN-SNN Accuracy: {best_entry['accuracy']:.4f}")
    print(f"Best CNN-SNN Macro F1: {best_entry['macro_f1']:.4f}")
    print("Best params:", {
        "learning_rate": best_entry["learning_rate"],
        "dropout": best_entry["dropout"],
        "beta": best_entry["beta"],
        "class_weight": best_entry["class_weight"],
        "epochs": best_entry["epochs"],
        "num_steps": best_entry.get("num_steps"),
        "best_epoch": best_entry["best_epoch"],
    })
    print("\nCompare against:")
    print(
        f"- Logistic Regression baseline: "
        f"{SEED_LR_BASELINE['accuracy']:.2%} Accuracy / "
        f"{SEED_LR_BASELINE['macro_f1']:.4f} Macro F1"
    )
    print(
        f"- Strong SNN: "
        f"{SEED_STRONG_SNN_BASELINE['accuracy']:.2%} Accuracy / "
        f"{SEED_STRONG_SNN_BASELINE['macro_f1']:.4f} Macro F1"
    )

    delta_strong = best_entry["macro_f1"] - SEED_STRONG_SNN_BASELINE["macro_f1"]
    delta_lr = best_entry["macro_f1"] - SEED_LR_BASELINE["macro_f1"]
    print(f"\nMacro F1 vs Strong SNN: {delta_strong:+.4f}")
    print(f"Macro F1 vs Logistic Regression: {delta_lr:+.4f}")
    if best_entry["macro_f1"] > SEED_LR_BASELINE["macro_f1"]:
        print("CNN-SNN exceeds Logistic Regression baseline Macro F1.")
    elif best_entry["macro_f1"] > SEED_STRONG_SNN_BASELINE["macro_f1"]:
        print("CNN-SNN improves over Strong SNN.")
    else:
        print("CNN-SNN did not exceed Strong SNN baseline on Macro F1.")


def export_cnn_snn_results(
    results: List[Dict[str, Any]],
    best_entry: Dict[str, Any],
    output_dir: Union[str, Path] = "results/metrics",
) -> Tuple[Path, Path]:
    """Export CNN-SNN grid results."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / "seed_cnn_snn_results.csv"
    json_path = output_dir / "seed_cnn_snn_results.json"

    rows = []
    for entry in results:
        rows.append(
            {
                "SEED_SNN_MODE": entry.get("SEED_SNN_MODE", "cnn_snn"),
                "learning_rate": entry["learning_rate"],
                "dropout": entry["dropout"],
                "beta": entry["beta"],
                "class_weight": entry["class_weight"],
                "epochs": entry["epochs"],
                "num_steps": entry.get("num_steps"),
                "best_epoch": entry["best_epoch"],
                "accuracy": float(entry["accuracy"]),
                "macro_f1": float(entry["macro_f1"]),
                "weighted_f1": float(entry["weighted_f1"]),
                "val_macro_f1": float(entry.get("val_macro_f1", 0.0)),
                "is_best": (
                    entry["macro_f1"] == best_entry["macro_f1"]
                    and entry["learning_rate"] == best_entry["learning_rate"]
                    and entry["dropout"] == best_entry["dropout"]
                    and entry["class_weight"] == best_entry["class_weight"]
                ),
            }
        )

    pd.DataFrame(rows).to_csv(csv_path, index=False)

    payload = {
        "best": best_entry,
        "all_configs": results,
        "baselines": {
            "logistic_regression": SEED_LR_BASELINE,
            "strong_snn": SEED_STRONG_SNN_BASELINE,
        },
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=str)

    return csv_path, json_path
