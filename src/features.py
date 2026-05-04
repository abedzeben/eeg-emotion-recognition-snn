from __future__ import annotations

import numpy as np
from scipy import signal


def flatten_time_series(X: np.ndarray) -> np.ndarray:
    """
    Convert (n_samples, n_channels, n_times) to (n_samples, n_channels * n_times).
    """
    if X.ndim != 3:
        raise ValueError(f"Expected X with shape (n_samples, n_channels, n_times), got {X.shape}")
    n, c, t = X.shape
    return X.reshape(n, c * t)


def bandpower_features(
    X: np.ndarray,
    *,
    sfreq: float,
    bands: list[tuple[float, float]] | None = None,
) -> np.ndarray:
    """
    Simple bandpower features per channel using Welch PSD.

    Returns: (n_samples, n_channels * n_bands)
    """
    if X.ndim != 3:
        raise ValueError(f"Expected X with shape (n_samples, n_channels, n_times), got {X.shape}")
    if bands is None:
        bands = [(4, 8), (8, 13), (13, 30), (30, 40)]  # theta/alpha/beta/low-gamma

    n_samples, n_channels, _ = X.shape
    feats = []
    for i in range(n_samples):
        f, Pxx = signal.welch(X[i], fs=sfreq, axis=-1, nperseg=min(256, X.shape[-1]))
        bp = []
        for (lo, hi) in bands:
            mask = (f >= lo) & (f <= hi)
            bp.append(np.trapz(Pxx[:, mask], f[mask], axis=-1))
        bp = np.stack(bp, axis=-1)  # (n_channels, n_bands)
        feats.append(bp.reshape(-1))
    return np.stack(feats, axis=0).astype(np.float32)


def extract_features(X: np.ndarray) -> np.ndarray:
    """
    Per-trial features from epoched EEG.

    X: (trials, channels, samples)
    For each trial and channel, compute mean and variance along the time axis,
    then flatten to one vector per trial: [ch0_mean, ch0_var, ch1_mean, ch1_var, ...].

    Returns:
        Feature matrix of shape (num_trials, num_features) with num_features = 2 * num_channels.
    """
    if X.ndim != 3:
        raise ValueError(f"Expected X with shape (trials, channels, samples), got {X.shape}")

    mean = np.mean(X, axis=-1)
    var = np.var(X, axis=-1)
    # (trials, channels, 2) -> (trials, 2 * channels)
    stacked = np.stack([mean, var], axis=-1)
    return stacked.reshape(X.shape[0], -1).astype(np.float32)
