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


def _band_average_powers(trial: np.ndarray, fs: float = 128.0) -> np.ndarray:
    """
    Average Welch PSD power per channel inside each frequency band.

    trial: (channels, samples)
    Returns: (channels, 4) for theta, alpha, beta, gamma.
    """
    bands = [(4, 8), (8, 13), (13, 30), (30, 45)]
    f, Pxx = signal.welch(trial, fs=fs, axis=-1, nperseg=min(256, trial.shape[-1]))
    band_powers = []
    for lo, hi in bands:
        mask = (f >= lo) & (f <= hi)
        band_powers.append(np.mean(Pxx[:, mask], axis=-1))
    return np.stack(band_powers, axis=-1)


def extract_features(X: np.ndarray, fs: float = 128.0) -> np.ndarray:
    """
    Per-trial features from epoched EEG.

    X: (trials, channels, samples)
    For each trial and channel, compute:
      mean, variance, and band-average power (theta, alpha, beta, gamma).

    Per channel (6 features): mean, variance, theta, alpha, beta, gamma.
    For DEAP (40 channels): (num_trials, 240).

    Returns:
        Feature matrix of shape (num_trials, num_channels * 6).
    """
    if X.ndim != 3:
        raise ValueError(f"Expected X with shape (trials, channels, samples), got {X.shape}")

    mean = np.mean(X, axis=-1)
    var = np.var(X, axis=-1)

    n_trials = X.shape[0]
    band_feats = np.stack([_band_average_powers(X[i], fs=fs) for i in range(n_trials)], axis=0)

    # (trials, channels, 6): mean, var, theta, alpha, beta, gamma
    stacked = np.stack(
        [
            mean,
            var,
            band_feats[:, :, 0],
            band_feats[:, :, 1],
            band_feats[:, :, 2],
            band_feats[:, :, 3],
        ],
        axis=-1,
    )
    return stacked.reshape(n_trials, -1).astype(np.float32)
