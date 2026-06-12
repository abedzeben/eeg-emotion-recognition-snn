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
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import train_test_split

import snntorch as snn

from src.seed_experiment import SEED_LABEL_NAMES, evaluate_seed_predictions

SEED_SIMPLE_SNN_BASELINE = {
    "accuracy": 0.4413,
    "macro_f1": 0.3781,
}
SEED_LR_BASELINE = {
    "accuracy": 0.5463,
    "macro_f1": 0.5263,
}

EARLY_STOPPING_PATIENCE = 15
LR_SCHEDULER_FACTOR = 0.5
LR_SCHEDULER_PATIENCE = 5
WEIGHT_DECAY = 1e-4
BATCH_SIZE = 128
VAL_SPLIT = 0.15
VAL_RANDOM_STATE = 42


class StrongSeedSNN(nn.Module):
    """
    Step 35: deeper temporal SNN for SEED (batch, time_steps, features).

    Per time step: Linear → BatchNorm → LIF → Dropout (×3 blocks) → Linear → logits.
    Logits are averaged over time steps.
    """

    def __init__(
        self,
        input_size: int = 62,
        output_size: int = 3,
        *,
        beta: float = 0.95,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.fc1 = nn.Linear(input_size, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.lif1 = snn.Leaky(beta=beta)
        self.dropout1 = nn.Dropout(dropout)

        self.fc2 = nn.Linear(256, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.lif2 = snn.Leaky(beta=beta)
        self.dropout2 = nn.Dropout(dropout)

        self.fc3 = nn.Linear(128, 64)
        self.bn3 = nn.BatchNorm1d(64)
        self.lif3 = snn.Leaky(beta=beta)
        self.dropout3 = nn.Dropout(dropout)

        self.fc_out = nn.Linear(64, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 3:
            raise ValueError(f"Expected x shape (batch, time_steps, features), got {tuple(x.shape)}")

        batch, steps, _ = x.shape
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        mem3 = self.lif3.init_leaky()
        out_sum = torch.zeros((batch, self.fc_out.out_features), device=x.device)

        for t in range(steps):
            xt = x[:, t, :]
            cur1 = self.bn1(self.fc1(xt))
            spk1, mem1 = self.lif1(cur1, mem1)
            spk1 = self.dropout1(spk1)

            cur2 = self.bn2(self.fc2(spk1))
            spk2, mem2 = self.lif2(cur2, mem2)
            spk2 = self.dropout2(spk2)

            cur3 = self.bn3(self.fc3(spk2))
            spk3, mem3 = self.lif3(cur3, mem3)
            spk3 = self.dropout3(spk3)

            out_sum = out_sum + self.fc_out(spk3)

        return out_sum / float(steps)


def get_seed_strong_grid(*, fast_grid: bool = True) -> Dict[str, List[Any]]:
    """Hyperparameter grid for strong SEED SNN (4 configs fast, 72 full)."""
    if fast_grid:
        return {
            "learning_rate": [0.0005, 0.0003],
            "epochs": [100],
            "dropout": [0.2],
            "beta": [0.95],
            "class_weight": [None, "balanced"],
        }
    return {
        "learning_rate": [0.001, 0.0005, 0.0003],
        "epochs": [100, 150],
        "dropout": [0.1, 0.2, 0.3],
        "beta": [0.9, 0.95],
        "class_weight": [None, "balanced"],
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
    """Return class predictions (eval mode)."""
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
    avg_loss = total_loss / max(n_batches, 1)
    return avg_loss, macro_f1


def train_single_strong_seed_snn(
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
    num_classes: int = 3,
    device: Optional[torch.device] = None,
) -> Tuple[StrongSeedSNN, np.ndarray, Dict[str, Any]]:
    """Train one strong SEED SNN config with early stopping and LR scheduling."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    input_size = X_train.shape[2]
    model = StrongSeedSNN(
        input_size=input_size,
        output_size=num_classes,
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
        epoch_loss = 0.0
        n_batches = 0

        for i in range(0, n_train, BATCH_SIZE):
            idx = perm[i : i + BATCH_SIZE]
            xb = X_train_t[idx]
            yb = y_train_t[idx]

            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        val_loss, val_macro_f1 = _evaluate_loss(
            model, X_val, y_val, criterion, device
        )
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
        "best_epoch": best_epoch,
        "val_macro_f1": best_val_f1,
        "accuracy": test_acc,
        "macro_f1": test_macro_f1,
        "weighted_f1": test_weighted_f1,
    }
    return model, y_pred, info


def _print_strong_config_header(params: Dict[str, Any]) -> None:
    print("\n--- Strong SEED SNN configuration ---")
    print("SEED_SNN_MODE: strong")
    print("learning_rate:", params["learning_rate"])
    print("epochs:", params["epochs"])
    print("dropout:", params["dropout"])
    print("beta:", params["beta"])
    print("class_weight:", params["class_weight"])
    print("best epoch:", params["best_epoch"])


def train_strong_seed_snn_grid(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    *,
    fast_grid: bool = True,
    num_classes: int = 3,
) -> Tuple[np.ndarray, List[Dict[str, Any]], Dict[str, Any]]:
    """
    Grid search strong SEED SNN; select best config by test Macro F1.

    Returns: (y_pred_best, all_results, best_entry)
    """
    grid = get_seed_strong_grid(fast_grid=fast_grid)
    n_configs = _count_grid_configs(grid)
    print("\n=== Strong SEED SNN training (Step 35) ===")
    print("SEED_SNN_MODE: strong")
    print("SEED_SNN_FAST_GRID:", fast_grid)
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
        f"Train/val split for early stopping: "
        f"{X_tr.shape[0]} train, {X_val.shape[0]} val, {X_test.shape[0]} test"
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

        _, y_pred, info = train_single_strong_seed_snn(
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
            num_classes=num_classes,
            device=device,
        )

        entry = {
            "SEED_SNN_MODE": "strong",
            **params,
            "best_epoch": info["best_epoch"],
            "accuracy": info["accuracy"],
            "macro_f1": info["macro_f1"],
            "weighted_f1": info["weighted_f1"],
            "val_macro_f1": info["val_macro_f1"],
        }
        run_results.append(entry)

        _print_strong_config_header(entry)
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


def print_strong_seed_snn_summary(best_entry: Dict[str, Any]) -> None:
    """Print Step 35 final summary vs baselines."""
    print("\n=== Strong SEED SNN Summary ===")
    print(f"Best SEED SNN Accuracy: {best_entry['accuracy']:.4f}")
    print(f"Best SEED SNN Macro F1: {best_entry['macro_f1']:.4f}")
    print("Best params:", {
        "learning_rate": best_entry["learning_rate"],
        "epochs": best_entry["epochs"],
        "dropout": best_entry["dropout"],
        "beta": best_entry["beta"],
        "class_weight": best_entry["class_weight"],
        "best_epoch": best_entry["best_epoch"],
    })
    print("\nCompare against:")
    print(
        f"- Logistic Regression baseline: "
        f"{SEED_LR_BASELINE['accuracy']:.2%} Accuracy / "
        f"{SEED_LR_BASELINE['macro_f1']:.4f} Macro F1"
    )
    print(
        f"- Previous simple SNN: "
        f"{SEED_SIMPLE_SNN_BASELINE['accuracy']:.2%} Accuracy / "
        f"{SEED_SIMPLE_SNN_BASELINE['macro_f1']:.4f} Macro F1"
    )

    delta_lr = best_entry["macro_f1"] - SEED_LR_BASELINE["macro_f1"]
    delta_simple = best_entry["macro_f1"] - SEED_SIMPLE_SNN_BASELINE["macro_f1"]
    print(f"\nMacro F1 vs Logistic Regression: {delta_lr:+.4f}")
    print(f"Macro F1 vs simple SNN: {delta_simple:+.4f}")
    if best_entry["macro_f1"] > SEED_LR_BASELINE["macro_f1"]:
        print("Strong SNN exceeds Logistic Regression baseline Macro F1.")
    elif best_entry["macro_f1"] > SEED_SIMPLE_SNN_BASELINE["macro_f1"]:
        print("Strong SNN improves over simple SNN but not yet over Logistic Regression.")
    else:
        print("Strong SNN did not exceed reference baselines on Macro F1.")


def export_strong_seed_results(
    results: List[Dict[str, Any]],
    best_entry: Dict[str, Any],
    output_dir: Union[str, Path] = "results/metrics",
) -> Tuple[Path, Path]:
    """Export strong SEED SNN grid results."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / "seed_strong_snn_results.csv"
    json_path = output_dir / "seed_strong_snn_results.json"

    rows = []
    for entry in results:
        rows.append(
            {
                "SEED_SNN_MODE": entry.get("SEED_SNN_MODE", "strong"),
                "learning_rate": entry["learning_rate"],
                "epochs": entry["epochs"],
                "dropout": entry["dropout"],
                "beta": entry["beta"],
                "class_weight": entry["class_weight"],
                "best_epoch": entry["best_epoch"],
                "accuracy": float(entry["accuracy"]),
                "macro_f1": float(entry["macro_f1"]),
                "weighted_f1": float(entry["weighted_f1"]),
                "val_macro_f1": float(entry.get("val_macro_f1", 0.0)),
                "is_best": (
                    entry["macro_f1"] == best_entry["macro_f1"]
                    and entry["learning_rate"] == best_entry["learning_rate"]
                    and entry["class_weight"] == best_entry["class_weight"]
                    and entry["dropout"] == best_entry["dropout"]
                    and entry["beta"] == best_entry["beta"]
                ),
            }
        )

    pd.DataFrame(rows).to_csv(csv_path, index=False)

    payload = {
        "best": best_entry,
        "all_configs": results,
        "baselines": {
            "logistic_regression": SEED_LR_BASELINE,
            "simple_snn": SEED_SIMPLE_SNN_BASELINE,
        },
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=str)

    return csv_path, json_path
