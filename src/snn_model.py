from __future__ import annotations

from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

import snntorch as snn


class SimpleSNN(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = 32, output_size: int = 2):
        super().__init__()
        # hidden_size kept for backward compatibility; network uses 64 -> 32 hidden layers.
        self.fc1 = nn.Linear(input_size, 64)
        self.lif1 = snn.Leaky(beta=0.95)
        self.fc2 = nn.Linear(64, 32)
        self.lif2 = snn.Leaky(beta=0.95)
        self.fc3 = nn.Linear(32, output_size)

    def forward(self, x: torch.Tensor, num_steps: int = 10) -> torch.Tensor:
        """
        x: (batch, input_size)
        Returns logits-like tensor (batch, output_size) based on averaged output activity.
        """
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        spk_sum = torch.zeros((x.shape[0], self.fc3.out_features), device=x.device)

        for _ in range(num_steps):
            cur1 = self.fc1(x)
            spk1, mem1 = self.lif1(cur1, mem1)
            cur2 = self.fc2(spk1)
            spk2, mem2 = self.lif2(cur2, mem2)
            out = self.fc3(spk2)
            spk_sum = spk_sum + out

        return spk_sum / float(num_steps)


def train_snn_model(
    X: np.ndarray,
    y: np.ndarray,
) -> Tuple[nn.Module, np.ndarray, np.ndarray, np.ndarray, float]:
    """
    Train a minimal SNN classifier using snntorch.

    Uses a deeper Linear -> Leaky -> Linear -> Leaky -> Linear architecture for binary classification.
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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X_train_t = torch.tensor(X_train, dtype=torch.float32, device=device)
    y_train_t = torch.tensor(y_train, dtype=torch.long, device=device)
    X_test_t = torch.tensor(X_test, dtype=torch.float32, device=device)
    y_test_t = torch.tensor(y_test, dtype=torch.long, device=device)

    class_counts = np.bincount(y_train, minlength=2)
    class_weights = len(y_train) / (2 * class_counts)
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32, device=device)

    model = SimpleSNN(input_size=X.shape[1], hidden_size=32, output_size=2).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    batch_size = 64
    epochs = 50
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

        avg_loss = epoch_loss / max(n_batches, 1)
        print(f"Epoch {epoch} | Loss: {avg_loss:.2f}")

    model.eval()
    with torch.no_grad():
        logits = model(X_test_t, num_steps=num_steps)
        preds = torch.argmax(logits, dim=1)
        y_pred = preds.cpu().numpy()
        acc = float(accuracy_score(y_test, y_pred))

    return model, X_test, y_test, y_pred, acc

