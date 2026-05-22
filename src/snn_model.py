from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import snntorch as snn


def rate_encode_features(X_tensor: torch.Tensor, num_steps: int = 10) -> torch.Tensor:
    """
    Rate-encode feature vectors into spike trains.

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

    def forward(self, spikes: torch.Tensor) -> torch.Tensor:
        """
        Process spike input over time.

        spikes: (num_steps, batch, features)
        Returns averaged output activity: (batch, output_size)
        """
        if spikes.dim() == 2:
            # Legacy static input: (batch, features) — treat as one step
            spikes = spikes.unsqueeze(0)

        num_steps, batch_size, _ = spikes.shape
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        out_sum = torch.zeros((batch_size, self.fc3.out_features), device=spikes.device)

        for t in range(num_steps):
            x = spikes[t]
            cur1 = self.fc1(x)
            spk1, mem1 = self.lif1(cur1, mem1)
            cur2 = self.fc2(spk1)
            spk2, mem2 = self.lif2(cur2, mem2)
            out_sum = out_sum + self.fc3(spk2)

        return out_sum / float(num_steps)


def train_snn_model(
    X: np.ndarray,
    y: np.ndarray,
) -> Tuple[nn.Module, np.ndarray, np.ndarray, np.ndarray, float, float, Dict[str, Any]]:
    """
    Train an SNN with rate-encoded spike input (Step 11 hyperparameters).

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

    hidden_size = 64
    learning_rate = 0.0005
    epochs = 100
    num_steps = 10
    best_params: Dict[str, Any] = {
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

    class_counts = np.bincount(y_train, minlength=2)
    class_weights = len(y_train) / (2 * class_counts)
    weight_tensor = torch.tensor(class_weights, dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(weight=weight_tensor)

    model = SimpleSNN(input_size=X_train_s.shape[1], hidden_size=hidden_size, output_size=2).to(device)
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
