from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import snntorch as snn


def rate_encode_features(X_tensor: torch.Tensor, num_steps: int = 10) -> torch.Tensor:
    """
    Rate-encode feature vectors into spike trains (Step 12).

    X_tensor: (samples, features)
    Returns: (num_steps, samples, features) with values in {0.0, 1.0}
    """
    x_min = X_tensor.min()
    x_max = X_tensor.max()
    X_normalized = (X_tensor - x_min) / (x_max - x_min + 1e-8)

    spike_trains = []
    for _ in range(num_steps):
        spikes = (torch.rand_like(X_normalized) < X_normalized).float()
        spike_trains.append(spikes.unsqueeze(0))

    return torch.cat(spike_trains, dim=0)


def fit_temporal_rate_encoding_stats(X_train: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Per-feature min/max for temporal rate encoding (fit on training data only).

    X_train: (trials, windows, features)
    """
    if X_train.ndim != 3:
        raise ValueError(f"Expected X_train shape (trials, windows, features), got {X_train.shape}")
    flat = X_train.reshape(-1, X_train.shape[2])
    return flat.min(axis=0).astype(np.float32), flat.max(axis=0).astype(np.float32)


def temporal_rate_encode_torch(
    x: torch.Tensor,
    feat_min: torch.Tensor,
    feat_max: torch.Tensor,
    encoding_steps: int,
) -> torch.Tensor:
    """
    Rate-encode windowed DE features into spike trains (Step 30).

    x: (batch, windows, features)
    Returns: (batch, windows, encoding_steps, features) with values in {0.0, 1.0}
    """
    probs = (x - feat_min) / (feat_max - feat_min + 1e-8)
    probs = probs.clamp(0.0, 1.0).unsqueeze(2).expand(-1, -1, encoding_steps, -1)
    return (torch.rand_like(probs) < probs).float()


def print_temporal_spike_encoding_info(
    n_trials: int,
    n_windows: int,
    n_features: int,
    *,
    encoding_steps: int,
) -> None:
    """Print Step 30 temporal spike encoding summary."""
    encoded_shape = (n_trials, n_windows, encoding_steps, n_features)
    print("\n=== Temporal spike encoding (Step 30) ===")
    print("Temporal spike encoding enabled")
    print("Encoding steps:", encoding_steps)
    print("Temporal encoded shape:", encoded_shape)
    print("SNN input shape:", encoded_shape)


class SimpleSNN(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
        second_hidden_size: int = 32,
        output_size: int = 2,
        beta: float = 0.95,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.lif1 = snn.Leaky(beta=beta)
        self.dropout1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_size, second_hidden_size)
        self.lif2 = snn.Leaky(beta=beta)
        self.dropout2 = nn.Dropout(dropout)
        self.fc3 = nn.Linear(second_hidden_size, output_size)

    def forward(
        self,
        x: torch.Tensor,
        num_steps: int = 10,
        *,
        temporal: bool = False,
        temporal_spike: bool = False,
    ) -> torch.Tensor:
        """
        Step 11 (static): x shape (batch, features) — same input repeated over time steps.
        Step 12 (spikes): x shape (num_steps, batch, features) — rate-encoded input.
        Step 27 (temporal): x shape (batch, time_steps, features) — one feature vector per window.
        Step 30 (temporal spikes): x shape (batch, windows, encoding_steps, features).
        """
        if temporal_spike:
            if x.dim() != 4:
                raise ValueError(
                    "Temporal spike mode expects x shape "
                    "(batch, windows, encoding_steps, features), "
                    f"got {tuple(x.shape)}"
                )
            batch, windows, enc_steps, _ = x.shape
            mem1 = self.lif1.init_leaky()
            mem2 = self.lif2.init_leaky()
            out_sum = torch.zeros((batch, self.fc3.out_features), device=x.device)
            total_steps = windows * enc_steps
            for w in range(windows):
                for e in range(enc_steps):
                    cur1 = self.fc1(x[:, w, e, :])
                    spk1, mem1 = self.lif1(cur1, mem1)
                    spk1 = self.dropout1(spk1)
                    cur2 = self.fc2(spk1)
                    spk2, mem2 = self.lif2(cur2, mem2)
                    spk2 = self.dropout2(spk2)
                    out_sum = out_sum + self.fc3(spk2)
            return out_sum / float(total_steps)

        if temporal:
            if x.dim() != 3:
                raise ValueError(
                    f"Temporal mode expects x shape (batch, time_steps, features), got {tuple(x.shape)}"
                )
            batch, steps, _ = x.shape
            mem1 = self.lif1.init_leaky()
            mem2 = self.lif2.init_leaky()
            out_sum = torch.zeros((batch, self.fc3.out_features), device=x.device)
            for t in range(steps):
                cur1 = self.fc1(x[:, t, :])
                spk1, mem1 = self.lif1(cur1, mem1)
                spk1 = self.dropout1(spk1)
                cur2 = self.fc2(spk1)
                spk2, mem2 = self.lif2(cur2, mem2)
                spk2 = self.dropout2(spk2)
                out_sum = out_sum + self.fc3(spk2)
            return out_sum / float(steps)

        if x.dim() == 3:
            spikes = x
            steps = spikes.shape[0]
            mem1 = self.lif1.init_leaky()
            mem2 = self.lif2.init_leaky()
            out_sum = torch.zeros((spikes.shape[1], self.fc3.out_features), device=spikes.device)
            for t in range(steps):
                cur1 = self.fc1(spikes[t])
                spk1, mem1 = self.lif1(cur1, mem1)
                spk1 = self.dropout1(spk1)
                cur2 = self.fc2(spk1)
                spk2, mem2 = self.lif2(cur2, mem2)
                spk2 = self.dropout2(spk2)
                out_sum = out_sum + self.fc3(spk2)
            return out_sum / float(steps)

        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        out_sum = torch.zeros((x.shape[0], self.fc3.out_features), device=x.device)
        for _ in range(num_steps):
            cur1 = self.fc1(x)
            spk1, mem1 = self.lif1(cur1, mem1)
            spk1 = self.dropout1(spk1)
            cur2 = self.fc2(spk1)
            spk2, mem2 = self.lif2(cur2, mem2)
            spk2 = self.dropout2(spk2)
            out_sum = out_sum + self.fc3(spk2)
        return out_sum / float(num_steps)


def _num_classes(y: np.ndarray) -> int:
    return int(len(np.unique(y)))


def _class_weight_tensor(
    y_train: np.ndarray, num_classes: int, device: torch.device
) -> torch.Tensor:
    class_counts = np.bincount(y_train, minlength=num_classes).astype(np.float32)
    class_counts = np.maximum(class_counts, 1.0)
    weights = len(y_train) / (num_classes * class_counts)
    return torch.tensor(weights, dtype=torch.float32, device=device)


def _macro_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }


def _scale_temporal_features(
    X_train: np.ndarray,
    X_test: np.ndarray,
    scaler: StandardScaler,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit StandardScaler on flattened window features and reshape back to 3D."""
    n_train, n_steps, n_features = X_train.shape
    X_train_s = scaler.fit_transform(X_train.reshape(-1, n_features)).reshape(
        n_train, n_steps, n_features
    )
    n_test = X_test.shape[0]
    X_test_s = scaler.transform(X_test.reshape(-1, n_features)).reshape(
        n_test, n_steps, n_features
    )
    return X_train_s.astype(np.float32), X_test_s.astype(np.float32)


def _train_single_snn(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    *,
    hidden_size: int,
    second_hidden_size: int,
    beta: float,
    dropout: float,
    num_steps: int,
    learning_rate: float,
    epochs: int,
    class_weight_mode: Optional[str],
    device: torch.device,
    num_classes: int,
    temporal: bool = False,
    temporal_spike_encoding: bool = False,
    encoding_steps: int = 10,
    feat_min: Optional[np.ndarray] = None,
    feat_max: Optional[np.ndarray] = None,
    verbose: bool = False,
) -> Tuple[nn.Module, np.ndarray, Dict[str, float]]:
    """Train one tuned SNN configuration (static, temporal, or temporal spike input)."""
    X_train_t = torch.tensor(X_train, dtype=torch.float32, device=device)
    y_train_t = torch.tensor(y_train, dtype=torch.long, device=device)
    X_test_t = torch.tensor(X_test, dtype=torch.float32, device=device)

    feat_min_t: Optional[torch.Tensor] = None
    feat_max_t: Optional[torch.Tensor] = None
    if temporal_spike_encoding:
        if feat_min is None or feat_max is None:
            raise ValueError("feat_min and feat_max are required for temporal spike encoding")
        feat_min_t = torch.tensor(feat_min, dtype=torch.float32, device=device)
        feat_max_t = torch.tensor(feat_max, dtype=torch.float32, device=device)

    if class_weight_mode == "balanced":
        weight_tensor = _class_weight_tensor(y_train, num_classes, device)
        criterion = nn.CrossEntropyLoss(weight=weight_tensor)
    else:
        criterion = nn.CrossEntropyLoss()

    input_size = X_train.shape[2] if temporal else X_train.shape[1]
    model = SimpleSNN(
        input_size=input_size,
        hidden_size=hidden_size,
        second_hidden_size=second_hidden_size,
        output_size=num_classes,
        beta=beta,
        dropout=dropout,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    batch_size = 64

    model.train()
    n_train = X_train_t.shape[0]
    for epoch in range(1, epochs + 1):
        epoch_loss = 0.0
        n_batches = 0
        perm = torch.randperm(n_train, device=device)
        for i in range(0, n_train, batch_size):
            idx = perm[i : i + batch_size]
            xb = X_train_t[idx]
            yb = y_train_t[idx]

            optimizer.zero_grad()
            if temporal_spike_encoding:
                assert feat_min_t is not None and feat_max_t is not None
                xb_spikes = temporal_rate_encode_torch(xb, feat_min_t, feat_max_t, encoding_steps)
                logits = model(xb_spikes, temporal_spike=True)
            elif temporal:
                logits = model(xb, temporal=True)
            else:
                logits = model(xb, num_steps=num_steps)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        if verbose:
            avg_loss = epoch_loss / max(n_batches, 1)
            print(f"Epoch {epoch} | Loss: {avg_loss:.2f}")

    model.eval()
    with torch.no_grad():
        if temporal_spike_encoding:
            assert feat_min_t is not None and feat_max_t is not None
            test_spikes = temporal_rate_encode_torch(
                X_test_t, feat_min_t, feat_max_t, encoding_steps
            )
            logits = model(test_spikes, temporal_spike=True)
        elif temporal:
            logits = model(X_test_t, temporal=True)
        else:
            logits = model(X_test_t, num_steps=num_steps)
        y_pred = torch.argmax(logits, dim=1).cpu().numpy()

    return model, y_pred, _macro_metrics(y_test, y_pred)


def _get_snn_hyperparameter_grid(*, fast_grid: bool) -> Dict[str, list]:
    """Return Step 26 SNN grid (full or reduced for fast experiments)."""
    if fast_grid:
        return {
            "hidden_size_options": [64, 128],
            "second_hidden_size_options": [32, 64],
            "learning_rates": [0.001, 0.0005],
            "epoch_options": [50],
            "class_weight_options": [None, "balanced"],
            "num_steps_options": [10, 20],
            "beta_options": [0.95],
            "dropout_options": [0.0, 0.2],
        }
    return {
        "hidden_size_options": [64, 128, 256],
        "second_hidden_size_options": [32, 64, 128],
        "learning_rates": [0.001, 0.0005],
        "epoch_options": [50, 100],
        "class_weight_options": [None, "balanced"],
        "num_steps_options": [10, 20],
        "beta_options": [0.9, 0.95],
        "dropout_options": [0.0, 0.2],
    }


BEST_TEMPORAL_SNN_CONFIG: Dict[str, Any] = {
    "hidden_size": 128,
    "second_hidden_size": 32,
    "beta": 0.95,
    "dropout": 0.2,
    "learning_rate": 0.0005,
    "epochs": 50,
    "class_weight": None,
}


def _get_temporal_snn_fine_tune_grid() -> Dict[str, list]:
    """Return Step 28 focused grid for temporal windowed SNN."""
    return {
        "hidden_size_options": [128, 256],
        "second_hidden_size_options": [32, 64],
        "learning_rates": [0.0005, 0.0003],
        "epoch_options": [50, 100, 150],
        "class_weight_options": [None, "balanced"],
        "beta_options": [0.9, 0.95],
        "dropout_options": [0.1, 0.2, 0.3],
    }


def _count_snn_grid_configs(grid: Dict[str, list]) -> int:
    total = 1
    for values in grid.values():
        total *= len(values)
    return total


def train_tuned_snn_model(
    X: np.ndarray,
    y: np.ndarray,
    *,
    snn_fast_grid: bool = True,
    temporal: bool = False,
    temporal_fine_tune: bool = False,
    use_best_temporal_config: bool = False,
    temporal_spike_encoding: bool = False,
    encoding_steps: int = 10,
) -> Tuple[nn.Module, np.ndarray, np.ndarray, np.ndarray, float, float, Dict[str, Any]]:
    """
    Step 26/27/28/29/30: hyperparameter-tuned SNN (grid search or fixed best config).

    temporal=True: X shape (trials, time_steps, features) — one DE vector per window.
    temporal_spike_encoding=True: rate-encode each window to (windows, encoding_steps, features).
    use_best_temporal_config=True: train single Step 28 best config (Step 29).
    temporal_fine_tune=True: use Step 28 focused grid (temporal only).
    temporal=False: X shape (trials, features) — static features repeated over num_steps.

    Returns:
        model, X_test, y_test, y_pred, accuracy, macro_f1, best_params
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    scaler = StandardScaler()
    if temporal:
        X_train_s, X_test_s = _scale_temporal_features(X_train, X_test, scaler)
    else:
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = _num_classes(y)
    print("Detected num_classes:", num_classes)
    if temporal:
        print("Temporal SNN input enabled (Step 27)")
        print("SNN training tensor shape:", X_train_s.shape)
        if temporal_spike_encoding:
            print("Temporal spike encoding enabled")
            print("Encoding steps:", encoding_steps)
            encoded_shape = (
                X_train_s.shape[0],
                X_train_s.shape[1],
                encoding_steps,
                X_train_s.shape[2],
            )
            print("Temporal encoded shape:", encoded_shape)
            print("SNN input shape:", encoded_shape)

    feat_min: Optional[np.ndarray] = None
    feat_max: Optional[np.ndarray] = None
    if temporal and temporal_spike_encoding:
        feat_min, feat_max = fit_temporal_rate_encoding_stats(X_train_s)

    def _train_kwargs() -> Dict[str, Any]:
        return {
            "temporal": temporal,
            "temporal_spike_encoding": temporal_spike_encoding,
            "encoding_steps": encoding_steps,
            "feat_min": feat_min,
            "feat_max": feat_max,
        }

    if temporal and use_best_temporal_config:
        print("Using best Temporal SNN config only")
        print("Skipping grid search")
        cfg = BEST_TEMPORAL_SNN_CONFIG
        num_steps = X_train_s.shape[1]
        model, y_pred, metrics = _train_single_snn(
            X_train_s,
            y_train,
            X_test_s,
            y_test,
            hidden_size=cfg["hidden_size"],
            second_hidden_size=cfg["second_hidden_size"],
            beta=cfg["beta"],
            dropout=cfg["dropout"],
            num_steps=num_steps,
            learning_rate=cfg["learning_rate"],
            epochs=cfg["epochs"],
            class_weight_mode=cfg["class_weight"],
            device=device,
            num_classes=num_classes,
            verbose=False,
            **_train_kwargs(),
        )
        mode_name = "temporal_step30" if temporal_spike_encoding else "temporal_step29"
        best_params = {
            "mode": mode_name,
            "use_best_temporal_config": True,
            "temporal": True,
            "temporal_spike_encoding": temporal_spike_encoding,
            "encoding_steps": encoding_steps if temporal_spike_encoding else None,
            "num_classes": num_classes,
            "hidden_size": cfg["hidden_size"],
            "second_hidden_size": cfg["second_hidden_size"],
            "beta": cfg["beta"],
            "dropout": cfg["dropout"],
            "learning_rate": cfg["learning_rate"],
            "epochs": cfg["epochs"],
            "class_weight": cfg["class_weight"],
            "time_steps": num_steps,
            "features_per_step": X_train_s.shape[2],
        }
        print(f"Selected tuned SNN params: {best_params}")
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"Macro F1: {metrics['macro_f1']:.4f}")
        return (
            model,
            X_test,
            y_test,
            y_pred,
            metrics["accuracy"],
            metrics["macro_f1"],
            best_params,
        )

    if temporal and temporal_fine_tune:
        grid = _get_temporal_snn_fine_tune_grid()
        print("TEMPORAL_SNN_FINE_TUNE enabled")
        print(f"Total Temporal SNN configurations: {_count_snn_grid_configs(grid)}")
        mode_key = "temporal_step28"
    elif temporal:
        grid = _get_snn_hyperparameter_grid(fast_grid=snn_fast_grid)
        if snn_fast_grid:
            print("SNN_FAST_GRID enabled")
        grid_count = _count_snn_grid_configs(grid) // len(
            _get_snn_hyperparameter_grid(fast_grid=snn_fast_grid)["num_steps_options"]
        )
        print(f"Total SNN configurations: {grid_count}")
        mode_key = "temporal_step27"
    else:
        grid = _get_snn_hyperparameter_grid(fast_grid=snn_fast_grid)
        if snn_fast_grid:
            print("SNN_FAST_GRID enabled")
        print(f"Total SNN configurations: {_count_snn_grid_configs(grid)}")
        mode_key = "tuned_step26"

    hidden_size_options = grid["hidden_size_options"]
    second_hidden_size_options = grid["second_hidden_size_options"]
    learning_rates = grid["learning_rates"]
    epoch_options = grid["epoch_options"]
    class_weight_options = grid["class_weight_options"]
    beta_options = grid["beta_options"]
    dropout_options = grid["dropout_options"]
    num_steps_options = (
        [X_train_s.shape[1]]
        if temporal
        else _get_snn_hyperparameter_grid(fast_grid=snn_fast_grid)["num_steps_options"]
    )

    best_model: Optional[nn.Module] = None
    best_pred: Optional[np.ndarray] = None
    best_params: Dict[str, Any] = {"mode": mode_key}
    best_macro_f1 = -1.0
    best_acc = 0.0

    for hidden_size in hidden_size_options:
        for second_hidden_size in second_hidden_size_options:
            for beta in beta_options:
                for dropout in dropout_options:
                    for num_steps in num_steps_options:
                        for lr in learning_rates:
                            for epochs in epoch_options:
                                for cw in class_weight_options:
                                    model, y_pred, metrics = _train_single_snn(
                                        X_train_s,
                                        y_train,
                                        X_test_s,
                                        y_test,
                                        hidden_size=hidden_size,
                                        second_hidden_size=second_hidden_size,
                                        beta=beta,
                                        dropout=dropout,
                                        num_steps=num_steps,
                                        learning_rate=lr,
                                        epochs=epochs,
                                        class_weight_mode=cw,
                                        device=device,
                                        num_classes=num_classes,
                                        verbose=False,
                                        temporal=temporal,
                                        temporal_spike_encoding=temporal_spike_encoding,
                                        encoding_steps=encoding_steps,
                                        feat_min=feat_min,
                                        feat_max=feat_max,
                                    )

                                    if temporal:
                                        print(
                                            f"SNN config hidden={hidden_size}, "
                                            f"second_hidden={second_hidden_size}, "
                                            f"beta={beta}, dropout={dropout}, "
                                            f"time_steps={num_steps}, lr={lr}, "
                                            f"epochs={epochs}, class_weight={cw} | "
                                            f"acc={metrics['accuracy']:.4f} "
                                            f"macro_f1={metrics['macro_f1']:.4f}"
                                        )
                                    else:
                                        print(
                                            f"SNN config hidden={hidden_size}, "
                                            f"second_hidden={second_hidden_size}, "
                                            f"beta={beta}, dropout={dropout}, "
                                            f"num_steps={num_steps}, lr={lr}, "
                                            f"epochs={epochs}, class_weight={cw} | "
                                            f"acc={metrics['accuracy']:.4f} "
                                            f"macro_f1={metrics['macro_f1']:.4f}"
                                        )

                                    if metrics["macro_f1"] > best_macro_f1:
                                        best_macro_f1 = metrics["macro_f1"]
                                        best_acc = metrics["accuracy"]
                                        best_model = model
                                        best_pred = y_pred
                                        best_params = {
                                            "mode": (
                                                "temporal_step30"
                                                if temporal_spike_encoding
                                                else mode_key
                                            ),
                                            "temporal": temporal,
                                            "temporal_spike_encoding": temporal_spike_encoding,
                                            "encoding_steps": (
                                                encoding_steps if temporal_spike_encoding else None
                                            ),
                                            "temporal_fine_tune": temporal_fine_tune,
                                            "snn_fast_grid": snn_fast_grid,
                                            "num_classes": num_classes,
                                            "hidden_size": hidden_size,
                                            "second_hidden_size": second_hidden_size,
                                            "beta": beta,
                                            "dropout": dropout,
                                            "num_steps": num_steps,
                                            "learning_rate": lr,
                                            "epochs": epochs,
                                            "class_weight": cw,
                                        }
                                        if temporal:
                                            best_params["time_steps"] = num_steps
                                            best_params["features_per_step"] = X_train_s.shape[2]

    assert best_model is not None and best_pred is not None
    print(f"Selected tuned SNN params: {best_params}")
    print(f"Accuracy: {best_acc:.4f}")
    print(f"Macro F1: {best_macro_f1:.4f}")

    return best_model, X_test, y_test, best_pred, best_acc, best_macro_f1, best_params


def train_spike_encoded_snn_model(
    X: np.ndarray,
    y: np.ndarray,
) -> Tuple[nn.Module, np.ndarray, np.ndarray, np.ndarray, float, float, Dict[str, Any]]:
    """
    Step 12: SNN with rate-encoded spike input (fixed Step 11 best hyperparameters).

    Returns:
        model, X_test, y_test, y_pred, accuracy, macro_f1, best_params
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = _num_classes(y)
    print("Detected num_classes:", num_classes)

    hidden_size = 64
    learning_rate = 0.0005
    epochs = 100
    num_steps = 10
    best_params: Dict[str, Any] = {
        "mode": "spike_encoded_step12",
        "num_classes": num_classes,
        "hidden_size": hidden_size,
        "learning_rate": learning_rate,
        "epochs": epochs,
        "num_steps": num_steps,
        "encoding": "rate",
        "class_weight": "balanced",
    }

    X_train_t = torch.tensor(X_train_s, dtype=torch.float32, device=device)
    y_train_t = torch.tensor(y_train, dtype=torch.long, device=device)
    X_test_t = torch.tensor(X_test_s, dtype=torch.float32, device=device)

    weight_tensor = _class_weight_tensor(y_train, num_classes, device)
    criterion = nn.CrossEntropyLoss(weight=weight_tensor)

    model = SimpleSNN(
        input_size=X_train_s.shape[1],
        hidden_size=hidden_size,
        second_hidden_size=32,
        output_size=num_classes,
        beta=0.95,
        dropout=0.0,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    batch_size = 64
    n_train = X_train_t.shape[0]

    model.train()
    for epoch in range(1, epochs + 1):
        epoch_loss = 0.0
        n_batches = 0
        perm = torch.randperm(n_train, device=device)
        for i in range(0, n_train, batch_size):
            idx = perm[i : i + batch_size]
            xb = X_train_t[idx]
            yb = y_train_t[idx]

            spikes = rate_encode_features(xb, num_steps=num_steps)
            optimizer.zero_grad()
            logits = model(spikes)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        if epoch % 10 == 0:
            avg_loss = epoch_loss / max(n_batches, 1)
            print(f"Epoch {epoch} | Loss: {avg_loss:.2f}")

    model.eval()
    with torch.no_grad():
        test_logits = []
        n_test = X_test_t.shape[0]
        for i in range(0, n_test, batch_size):
            xb = X_test_t[i : i + batch_size]
            spikes = rate_encode_features(xb, num_steps=num_steps)
            test_logits.append(model(spikes))
        logits = torch.cat(test_logits, dim=0)
        y_pred = torch.argmax(logits, dim=1).cpu().numpy()

    acc = float(accuracy_score(y_test, y_pred))
    macro_f1 = float(f1_score(y_test, y_pred, average="macro", zero_division=0))

    return model, X_test, y_test, y_pred, acc, macro_f1, best_params


# Backward-compatible alias (Step 12 spike-encoded trainer)
train_snn_model = train_spike_encoded_snn_model
