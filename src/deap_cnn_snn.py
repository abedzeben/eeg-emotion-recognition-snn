from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

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
from sklearn.preprocessing import StandardScaler

import snntorch as snn

from src.evaluate import evaluate_classification
from src.features import TEMPORAL_NUM_WINDOWS, extract_temporal_window_de_features
from src.labels import EMOTION_LABELS, create_multi_emotion_labels
from src.snn_model import BEST_TEMPORAL_SNN_CONFIG, _scale_temporal_features, _train_single_snn

DEAP_NORMALIZATION_MODES = (
    "none",
    "global",
    "per_subject",
    "per_subject_per_channel",
)
DeapNormalizationMode = Literal[
    "none",
    "global",
    "per_subject",
    "per_subject_per_channel",
]

DEAP_TEMPORAL_BASELINE = {
    "accuracy": 0.5312,
    "macro_f1": 0.5103,
}

DEAP_CNN_SNN_CONFIG: Dict[str, Any] = {
    "learning_rate": 0.001,
    "dropout": 0.3,
    "beta": 0.95,
    "class_weight": "balanced",
    "epochs": 100,
}

WEIGHT_DECAY = 1e-4
BATCH_SIZE = 128
VAL_SPLIT = 0.15
VAL_RANDOM_STATE = 42
EARLY_STOPPING_PATIENCE = 15
LR_SCHEDULER_FACTOR = 0.5
LR_SCHEDULER_PATIENCE = 5
_SPLIT_RANDOM_STATE = 42
_EPS = 1e-8
NUM_BANDS = 5


class DeapTemporalCnnSnn(nn.Module):
    """
    Step 39: per-window CNN on (bands × channels) + temporal SNN aggregation.

    Input: (batch, windows, bands, channels)
    """

    def __init__(
        self,
        *,
        num_bands: int = 5,
        num_channels: int = 40,
        num_classes: int = 4,
        beta: float = 0.95,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.num_classes = num_classes

        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc_cnn = nn.Linear(32, 128)
        self.dropout_cnn = nn.Dropout(dropout)

        self.fc1 = nn.Linear(128, 128)
        self.lif1 = snn.Leaky(beta=beta)
        self.dropout1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(128, 64)
        self.lif2 = snn.Leaky(beta=beta)
        self.dropout2 = nn.Dropout(dropout)
        self.fc_out = nn.Linear(64, num_classes)

    def _extract_cnn_features(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, bands, channels) -> (batch, 128)"""
        if x.dim() != 3:
            raise ValueError(f"Expected (batch, bands, channels), got {tuple(x.shape)}")
        x_map = x.unsqueeze(1)
        x_map = F.relu(self.bn1(self.conv1(x_map)))
        x_map = F.relu(self.bn2(self.conv2(x_map)))
        x_map = self.adaptive_pool(x_map).flatten(1)
        feats = F.relu(self.fc_cnn(x_map))
        return self.dropout_cnn(feats)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 4:
            raise ValueError(f"Expected (batch, windows, bands, channels), got {tuple(x.shape)}")

        batch, windows, _, _ = x.shape
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        out_sum = torch.zeros((batch, self.num_classes), device=x.device)

        for w in range(windows):
            feats = self._extract_cnn_features(x[:, w])
            cur1 = self.fc1(feats)
            spk1, mem1 = self.lif1(cur1, mem1)
            spk1 = self.dropout1(spk1)
            cur2 = self.fc2(spk1)
            spk2, mem2 = self.lif2(cur2, mem2)
            spk2 = self.dropout2(spk2)
            out_sum = out_sum + self.fc_out(spk2)

        return out_sum / float(windows)


def reshape_temporal_de_to_band_channel(
    X_temporal: np.ndarray,
    *,
    num_bands: int = NUM_BANDS,
) -> np.ndarray:
    """
    (trials, windows, channels * bands) -> (trials, windows, bands, channels).

    DE storage order: channel-major (ch0 bands, ch1 bands, ...).
    """
    if X_temporal.ndim != 3:
        raise ValueError(f"Expected (trials, windows, features), got {X_temporal.shape}")
    n_trials, n_windows, flat = X_temporal.shape
    if flat % num_bands != 0:
        raise ValueError(f"Feature size {flat} is not divisible by {num_bands} bands")
    n_channels = flat // num_bands
    return (
        X_temporal.reshape(n_trials, n_windows, n_channels, num_bands)
        .transpose(0, 1, 3, 2)
        .astype(np.float32, copy=False)
    )


def flatten_band_channel_to_temporal(X_4d: np.ndarray) -> np.ndarray:
    """(trials, windows, bands, channels) -> (trials, windows, channels * bands)."""
    n_trials, n_windows, n_bands, n_channels = X_4d.shape
    return (
        X_4d.transpose(0, 1, 3, 2)
        .reshape(n_trials, n_windows, n_channels * n_bands)
        .astype(np.float32, copy=False)
    )


def normalize_deap_temporal_4d(
    X_train: np.ndarray,
    X_test: np.ndarray,
    subjects_train: np.ndarray,
    subjects_test: np.ndarray,
    mode: DeapNormalizationMode,
) -> Tuple[np.ndarray, np.ndarray]:
    """Normalize (trials, windows, bands, channels); train stats only where applicable."""
    if mode not in DEAP_NORMALIZATION_MODES:
        valid = ", ".join(DEAP_NORMALIZATION_MODES)
        raise ValueError(f"Unknown DEAP normalization mode '{mode}'. Valid: {valid}")

    if mode == "none":
        return X_train.astype(np.float32, copy=True), X_test.astype(np.float32, copy=True)

    _, _, n_bands, n_channels = X_train.shape

    if mode == "global":
        scaler = StandardScaler()
        scaler.fit(X_train.reshape(-1, n_channels))
        X_train_n = scaler.transform(X_train.reshape(-1, n_channels)).reshape(X_train.shape)
        X_test_n = scaler.transform(X_test.reshape(-1, n_channels)).reshape(X_test.shape)
        return X_train_n.astype(np.float32), X_test_n.astype(np.float32)

    out_train = np.empty_like(X_train, dtype=np.float32)
    out_test = np.empty_like(X_test, dtype=np.float32)

    if mode == "per_subject":
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
        stats: Dict[Tuple[int, int], Tuple[float, float]] = {}
        for subj in np.unique(subjects_train):
            mask = subjects_train == subj
            for ch in range(n_channels):
                block = X_train[mask, :, :, ch]
                mean = float(np.mean(block))
                std = float(np.std(block)) + _EPS
                stats[(int(subj), ch)] = (mean, std)
                out_train[mask, :, :, ch] = (block - mean) / std

        global_mean = float(np.mean(X_train))
        global_std = float(np.std(X_train)) + _EPS

        for subj in np.unique(subjects_test):
            mask = subjects_test == subj
            for ch in range(n_channels):
                block = X_test[mask, :, :, ch]
                if (int(subj), ch) in stats:
                    mean, std = stats[(int(subj), ch)]
                else:
                    mean, std = global_mean, global_std
                out_test[mask, :, :, ch] = (block - mean) / std
        return out_train, out_test

    raise ValueError(f"Unhandled normalization mode: {mode}")


def _class_weight_tensor(
    y_train: np.ndarray,
    num_classes: int,
    device: torch.device,
) -> torch.Tensor:
    class_counts = np.bincount(y_train, minlength=num_classes).astype(np.float32)
    class_counts = np.maximum(class_counts, 1.0)
    weights = len(y_train) / (num_classes * class_counts)
    return torch.tensor(weights, dtype=torch.float32, device=device)


def _predict_cnn_snn(
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
            logits = model(X_t[i : i + batch_size])
            preds.append(torch.argmax(logits, dim=1).cpu().numpy())
    return np.concatenate(preds, axis=0)


def _evaluate_cnn_snn_val(
    model: nn.Module,
    X: np.ndarray,
    y: np.ndarray,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    model.eval()
    X_t = torch.tensor(X, dtype=torch.float32, device=device)
    y_t = torch.tensor(y, dtype=torch.long, device=device)
    total_loss = 0.0
    n_batches = 0
    all_preds: List[np.ndarray] = []
    with torch.no_grad():
        for i in range(0, X_t.shape[0], BATCH_SIZE):
            xb = X_t[i : i + BATCH_SIZE]
            yb = y_t[i : i + BATCH_SIZE]
            logits = model(xb)
            total_loss += criterion(logits, yb).item()
            n_batches += 1
            all_preds.append(torch.argmax(logits, dim=1).cpu().numpy())
    y_pred = np.concatenate(all_preds, axis=0)
    macro_f1 = float(f1_score(y, y_pred, average="macro", zero_division=0))
    return total_loss / max(n_batches, 1), macro_f1


def train_deap_cnn_snn(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    *,
    num_classes: int = 4,
    config: Optional[Dict[str, Any]] = None,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Train DEAP CNN-SNN on pre-split 4D data."""
    cfg = {**DEAP_CNN_SNN_CONFIG, **(config or {})}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train,
        y_train,
        test_size=VAL_SPLIT,
        random_state=VAL_RANDOM_STATE,
        stratify=y_train,
    )

    if cfg.get("class_weight") == "balanced":
        weight_tensor = _class_weight_tensor(y_tr, num_classes, device)
        criterion = nn.CrossEntropyLoss(weight=weight_tensor)
    else:
        criterion = nn.CrossEntropyLoss()

    model = DeapTemporalCnnSnn(
        num_bands=X_train.shape[2],
        num_channels=X_train.shape[3],
        num_classes=num_classes,
        beta=float(cfg["beta"]),
        dropout=float(cfg["dropout"]),
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg["learning_rate"]),
        weight_decay=WEIGHT_DECAY,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=LR_SCHEDULER_FACTOR,
        patience=LR_SCHEDULER_PATIENCE,
    )

    X_tr_t = torch.tensor(X_tr, dtype=torch.float32, device=device)
    y_tr_t = torch.tensor(y_tr, dtype=torch.long, device=device)

    best_state: Optional[Dict[str, torch.Tensor]] = None
    best_val_f1 = -1.0
    best_epoch = 0
    patience_counter = 0
    max_epochs = int(cfg["epochs"])

    model.train()
    n_train = X_tr_t.shape[0]
    for epoch in range(1, max_epochs + 1):
        perm = torch.randperm(n_train, device=device)
        for i in range(0, n_train, BATCH_SIZE):
            idx = perm[i : i + BATCH_SIZE]
            optimizer.zero_grad()
            logits = model(X_tr_t[idx])
            loss = criterion(logits, y_tr_t[idx])
            loss.backward()
            optimizer.step()

        val_loss, val_f1 = _evaluate_cnn_snn_val(model, X_val, y_val, criterion, device)
        scheduler.step(val_loss)

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
        if patience_counter >= EARLY_STOPPING_PATIENCE:
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    y_pred = _predict_cnn_snn(model, X_test, device)
    info = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "macro_f1": float(f1_score(y_test, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_test, y_pred, average="weighted", zero_division=0)),
        "best_epoch": best_epoch,
        "config": cfg,
    }
    return y_pred, info


def train_deap_temporal_baseline(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    *,
    num_classes: int = 4,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Train existing best DEAP Temporal SNN on (trials, windows, features)."""
    cfg = BEST_TEMPORAL_SNN_CONFIG
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    scaler = StandardScaler()
    X_train_s, X_test_s = _scale_temporal_features(X_train, X_test, scaler)

    if cfg.get("class_weight") == "balanced":
        class_weight_mode = "balanced"
    else:
        class_weight_mode = cfg.get("class_weight")

    _, y_pred, metrics = _train_single_snn(
        X_train_s,
        y_train,
        X_test_s,
        y_test,
        hidden_size=cfg["hidden_size"],
        second_hidden_size=cfg["second_hidden_size"],
        beta=cfg["beta"],
        dropout=cfg["dropout"],
        num_steps=X_train_s.shape[1],
        learning_rate=cfg["learning_rate"],
        epochs=cfg["epochs"],
        class_weight_mode=class_weight_mode,
        device=device,
        num_classes=num_classes,
        temporal=True,
    )
    info = {
        "accuracy": metrics["accuracy"],
        "macro_f1": metrics["macro_f1"],
        "weighted_f1": float(f1_score(y_test, y_pred, average="weighted", zero_division=0)),
        "config": cfg,
    }
    return y_pred, info


def _metrics_dict(y_test: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "macro_f1": float(f1_score(y_test, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_test, y_pred, average="weighted", zero_division=0)),
    }


def export_deap_cnn_snn_results(
    results: List[Dict[str, Any]],
    *,
    output_dir: Union[str, Path] = "results/metrics",
) -> Tuple[Path, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / "deap_cnn_snn_results.csv"
    json_path = output_dir / "deap_cnn_snn_results.json"

    rows = []
    for entry in results:
        rows.append(
            {
                "model": entry["model"],
                "normalization": entry.get("normalization", ""),
                "accuracy": float(entry["accuracy"]),
                "macro_f1": float(entry["macro_f1"]),
                "weighted_f1": float(entry["weighted_f1"]),
            }
        )
    pd.DataFrame(rows).to_csv(csv_path, index=False)

    payload = {
        "study": "Step 39 DEAP CNN-SNN",
        "DEAP_NORMALIZATION_MODE": results[0].get("normalization") if results else "",
        "reference_baseline": DEAP_TEMPORAL_BASELINE,
        "models": results,
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=str)

    return csv_path, json_path


def run_deap_cnn_snn_experiment(
    *,
    folder: str = "data/raw",
    max_subjects: Optional[int] = None,
    normalization_mode: DeapNormalizationMode = "per_subject_per_channel",
    label_strategy: str = "mean",
    trials_per_subject: int = 40,
) -> List[Dict[str, Any]]:
    """Step 39: compare DEAP Temporal SNN vs CNN-SNN with subject-aware normalization."""
    from src.load_data import load_all_deap_files
    from src.preprocessing import bandpass_filter

    print("\n" + "=" * 60)
    print("DEAP CNN-SNN Experiment (Step 39)")
    print("=" * 60)
    print("DEAP_NORMALIZATION_MODE:", normalization_mode)
    print("Label strategy:", label_strategy)

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
    print("\nOriginal temporal shape:", X_temporal.shape)

    X_4d = reshape_temporal_de_to_band_channel(X_temporal)
    print("Reshaped temporal shape:", X_4d.shape)

    y_multi = create_multi_emotion_labels(y_ratings, strategy=label_strategy, verbose=False)
    print("y_multi shape:", y_multi.shape)

    (
        X_train,
        X_test,
        y_train,
        y_test,
        subj_train,
        subj_test,
    ) = train_test_split(
        X_4d,
        y_multi,
        subject_ids,
        test_size=0.2,
        random_state=_SPLIT_RANDOM_STATE,
        stratify=y_multi,
    )

    X_train_n, X_test_n = normalize_deap_temporal_4d(
        X_train,
        X_test,
        subj_train,
        subj_test,
        normalization_mode,
    )

    X_train_flat = flatten_band_channel_to_temporal(X_train_n)
    X_test_flat = flatten_band_channel_to_temporal(X_test_n)

    print(f"\nTrain samples: {X_train_n.shape[0]}, Test samples: {X_test_n.shape[0]}")

    print("\n--- Training DEAP Temporal SNN (baseline) ---")
    baseline_pred, baseline_info = train_deap_temporal_baseline(
        X_train_flat,
        y_train,
        X_test_flat,
        y_test,
        num_classes=4,
    )
    evaluate_classification(
        y_test,
        baseline_pred,
        "DEAP Temporal SNN (baseline)",
        num_classes=4,
    )

    print("\n--- Training DEAP CNN-SNN ---")
    cnn_pred, cnn_info = train_deap_cnn_snn(
        X_train_n,
        y_train,
        X_test_n,
        y_test,
        num_classes=4,
    )
    evaluate_classification(
        y_test,
        cnn_pred,
        "DEAP CNN-SNN",
        num_classes=4,
    )

    baseline_metrics = _metrics_dict(y_test, baseline_pred)
    cnn_metrics = _metrics_dict(y_test, cnn_pred)

    print("\n=== DEAP Comparison ===")
    print("Model | Accuracy | Macro F1")
    print(
        f"Temporal SNN | {baseline_metrics['accuracy']:.4f} | "
        f"{baseline_metrics['macro_f1']:.4f}"
    )
    print(
        f"CNN-SNN | {cnn_metrics['accuracy']:.4f} | "
        f"{cnn_metrics['macro_f1']:.4f}"
    )

    acc_delta = cnn_metrics["accuracy"] - baseline_metrics["accuracy"]
    f1_delta = cnn_metrics["macro_f1"] - baseline_metrics["macro_f1"]
    print("\nImprovement (CNN-SNN vs Temporal SNN):")
    print(f"Accuracy delta: {acc_delta:+.4f}")
    print(f"Macro F1 delta: {f1_delta:+.4f}")

    ref_acc_delta = cnn_metrics["accuracy"] - DEAP_TEMPORAL_BASELINE["accuracy"]
    ref_f1_delta = cnn_metrics["macro_f1"] - DEAP_TEMPORAL_BASELINE["macro_f1"]
    print("\nCNN-SNN vs historical DEAP baseline (~53.12% / 0.5103):")
    print(f"Accuracy delta: {ref_acc_delta:+.4f}")
    print(f"Macro F1 delta: {ref_f1_delta:+.4f}")

    results = [
        {
            "model": "Temporal SNN",
            "normalization": normalization_mode,
            **baseline_metrics,
            "params": baseline_info.get("config"),
        },
        {
            "model": "CNN-SNN",
            "normalization": normalization_mode,
            **cnn_metrics,
            "params": cnn_info.get("config"),
            "best_epoch": cnn_info.get("best_epoch"),
        },
    ]

    csv_path, json_path = export_deap_cnn_snn_results(results)
    print("\nDEAP CNN-SNN results saved:")
    print(" ", csv_path)
    print(" ", json_path)

    return results


def run_final_dataset_comparison(
    *,
    seed_results_path: Union[str, Path] = "results/metrics/seed_best_cnn_snn_results.json",
    deap_results_path: Union[str, Path] = "results/metrics/deap_cnn_snn_results.json",
) -> None:
    """Step 39: compare saved SEED and DEAP CNN-SNN results (no retraining)."""
    seed_path = Path(seed_results_path)
    deap_path = Path(deap_results_path)

    if not seed_path.exists():
        raise FileNotFoundError(f"SEED results not found: {seed_path}")
    if not deap_path.exists():
        raise FileNotFoundError(f"DEAP results not found: {deap_path}")

    with open(seed_path, encoding="utf-8") as f:
        seed_data = json.load(f)
    with open(deap_path, encoding="utf-8") as f:
        deap_data = json.load(f)

    seed_metrics = seed_data.get("metrics", seed_data)
    deap_models = deap_data.get("models", deap_data)
    if isinstance(deap_models, list):
        deap_cnn = next(
            (m for m in deap_models if m.get("model") == "CNN-SNN"),
            deap_models[-1],
        )
    else:
        deap_cnn = deap_models

    seed_acc = float(seed_metrics.get("accuracy", 0.0))
    seed_f1 = float(seed_metrics.get("macro_f1", 0.0))
    deap_acc = float(deap_cnn.get("accuracy", 0.0))
    deap_f1 = float(deap_cnn.get("macro_f1", 0.0))

    print("\n=== Final Dataset Comparison ===")
    print("Dataset | Accuracy | Macro F1")
    print(f"DEAP CNN-SNN | {deap_acc:.4f} | {deap_f1:.4f}")
    print(f"SEED CNN-SNN | {seed_acc:.4f} | {seed_f1:.4f}")

    if seed_f1 >= deap_f1 and seed_acc >= deap_acc:
        best_dataset = "SEED"
        best_acc = seed_acc
        best_f1 = seed_f1
    elif deap_f1 > seed_f1:
        best_dataset = "DEAP"
        best_acc = deap_acc
        best_f1 = deap_f1
    else:
        best_dataset = "SEED"
        best_acc = seed_acc
        best_f1 = seed_f1

    print("\nBest Dataset:", best_dataset)
    print(f"Best Accuracy: {best_acc:.4f}")
    print(f"Best Macro F1: {best_f1:.4f}")
