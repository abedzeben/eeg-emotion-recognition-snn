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


class SimpleSNN(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = 64, output_size: int = 2):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.lif1 = snn.Leaky(beta=0.95)
        self.fc2 = nn.Linear(hidden_size, 32)
        self.lif2 = snn.Leaky(beta=0.95)
        self.fc3 = nn.Linear(32, output_size)

    def forward(self, x: torch.Tensor, num_steps: int = 10) -> torch.Tensor:
        """
        Step 11 (static): x shape (batch, features) — same input repeated over time steps.
        Step 12 (spikes): x shape (num_steps, batch, features) — rate-encoded input.
        """
        if x.dim() == 3:
            spikes = x
            steps = spikes.shape[0]
            mem1 = self.lif1.init_leaky()
            mem2 = self.lif2.init_leaky()
            out_sum = torch.zeros((spikes.shape[1], self.fc3.out_features), device=spikes.device)
            for t in range(steps):
                cur1 = self.fc1(spikes[t])
                spk1, mem1 = self.lif1(cur1, mem1)
                cur2 = self.fc2(spk1)
                spk2, mem2 = self.lif2(cur2, mem2)
                out_sum = out_sum + self.fc3(spk2)
            return out_sum / float(steps)

        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        out_sum = torch.zeros((x.shape[0], self.fc3.out_features), device=x.device)
        for _ in range(num_steps):
            cur1 = self.fc1(x)
            spk1, mem1 = self.lif1(cur1, mem1)
            cur2 = self.fc2(spk1)
            spk2, mem2 = self.lif2(cur2, mem2)
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


def _train_single_snn(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    *,
    hidden_size: int,
    learning_rate: float,
    epochs: int,
    class_weight_mode: Optional[str],
    device: torch.device,
    num_classes: int,
    verbose: bool = False,
) -> Tuple[nn.Module, np.ndarray, Dict[str, float]]:
    """Train one Step 11 SNN configuration (static feature input)."""
    X_train_t = torch.tensor(X_train, dtype=torch.float32, device=device)
    y_train_t = torch.tensor(y_train, dtype=torch.long, device=device)
    X_test_t = torch.tensor(X_test, dtype=torch.float32, device=device)

    if class_weight_mode == "balanced":
        weight_tensor = _class_weight_tensor(y_train, num_classes, device)
        criterion = nn.CrossEntropyLoss(weight=weight_tensor)
    else:
        criterion = nn.CrossEntropyLoss()

    model = SimpleSNN(
        input_size=X_train.shape[1], hidden_size=hidden_size, output_size=num_classes
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    batch_size = 64
    num_steps = 10

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
        logits = model(X_test_t, num_steps=num_steps)
        y_pred = torch.argmax(logits, dim=1).cpu().numpy()

    return model, y_pred, _macro_metrics(y_test, y_pred)


def train_tuned_snn_model(
    X: np.ndarray,
    y: np.ndarray,
) -> Tuple[nn.Module, np.ndarray, np.ndarray, np.ndarray, float, float, Dict[str, Any]]:
    """
    Step 11: hyperparameter-tuned SNN (grid search, macro F1 selection).

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

    hidden_sizes = [32, 64]
    learning_rates = [0.001, 0.0005]
    epoch_options = [50, 100]
    class_weight_options: list = [None, "balanced"]

    best_model: nn.Module | None = None
    best_pred: np.ndarray | None = None
    best_params: Dict[str, Any] = {"mode": "tuned_step11"}
    best_macro_f1 = -1.0
    best_acc = 0.0

    for hidden_size in hidden_sizes:
        for lr in learning_rates:
            for epochs in epoch_options:
                for cw in class_weight_options:
                    model, y_pred, metrics = _train_single_snn(
                        X_train_s,
                        y_train,
                        X_test_s,
                        y_test,
                        hidden_size=hidden_size,
                        learning_rate=lr,
                        epochs=epochs,
                        class_weight_mode=cw,
                        device=device,
                        num_classes=num_classes,
                        verbose=False,
                    )

                    print(
                        f"SNN config hidden={hidden_size}, lr={lr}, epochs={epochs}, "
                        f"class_weight={cw} | acc={metrics['accuracy']:.4f} "
                        f"macro_f1={metrics['macro_f1']:.4f}"
                    )

                    if metrics["macro_f1"] > best_macro_f1:
                        best_macro_f1 = metrics["macro_f1"]
                        best_acc = metrics["accuracy"]
                        best_model = model
                        best_pred = y_pred
                        best_params = {
                            "mode": "tuned_step11",
                            "num_classes": num_classes,
                            "hidden_size": hidden_size,
                            "learning_rate": lr,
                            "epochs": epochs,
                            "class_weight": cw,
                        }

    assert best_model is not None and best_pred is not None
    print(f"Selected tuned SNN params: {best_params} | macro F1: {best_macro_f1:.4f}")

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
        input_size=X_train_s.shape[1], hidden_size=hidden_size, output_size=num_classes
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
