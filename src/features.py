from __future__ import annotations

import numpy as np
from scipy import signal

# Step 19 frequency bands (Welch PSD average power per band)
FREQUENCY_BANDS: list[tuple[float, float]] = [
    (0.5, 4),   # Delta
    (4, 8),     # Theta
    (8, 13),    # Alpha
    (13, 30),   # Beta
    (30, 45),   # Gamma
]


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


def _extract_statistical_features_six(X: np.ndarray) -> np.ndarray:
    """
    Six statistical features per channel.

    Returns: (trials, channels, 6) — mean, std, variance, min, max, median.
    """
    return np.stack(
        [
            np.mean(X, axis=-1),
            np.std(X, axis=-1),
            np.var(X, axis=-1),
            np.min(X, axis=-1),
            np.max(X, axis=-1),
            np.median(X, axis=-1),
        ],
        axis=-1,
    )


def _welch_band_powers(
    trial: np.ndarray,
    fs: float = 128.0,
    bands: list[tuple[float, float]] | None = None,
) -> np.ndarray:
    """
    Average Welch PSD power per channel inside each frequency band.

    trial: (channels, samples)
    Returns: (channels, n_bands)
    """
    if bands is None:
        bands = FREQUENCY_BANDS
    f, Pxx = signal.welch(trial, fs=fs, axis=-1, nperseg=min(256, trial.shape[-1]))
    band_powers = []
    for lo, hi in bands:
        mask = (f >= lo) & (f <= hi)
        band_powers.append(np.mean(Pxx[:, mask], axis=-1))
    return np.stack(band_powers, axis=-1)


def _extract_features_legacy(X: np.ndarray, fs: float = 128.0) -> np.ndarray:
    """
    Legacy feature extraction: statistical features only.

    Per channel (6 features): mean, std, variance, min, max, median.
    For DEAP (40 channels): (num_trials, 240).
    """
    n_trials = X.shape[0]
    stat_feats = _extract_statistical_features_six(X)
    return stat_feats.reshape(n_trials, -1).astype(np.float32)


def _extract_features_with_frequency(X: np.ndarray, fs: float = 128.0) -> np.ndarray:
    """
    Step 19: statistical features + Welch band power features.

    Per channel: 6 statistical + 5 band powers = 11 features.
    For DEAP (40 channels): (num_trials, 440).
    """
    n_trials, n_channels, _ = X.shape
    stat_feats = _extract_statistical_features_six(X)
    band_feats = np.stack(
        [_welch_band_powers(X[i], fs=fs, bands=FREQUENCY_BANDS) for i in range(n_trials)],
        axis=0,
    )
    stat_flat = stat_feats.reshape(n_trials, n_channels * 6)
    band_flat = band_feats.reshape(n_trials, n_channels * len(FREQUENCY_BANDS))
    return np.concatenate([stat_flat, band_flat], axis=1).astype(np.float32)


def extract_features(
    X: np.ndarray,
    fs: float = 128.0,
    *,
    use_frequency_features: bool = False,
) -> np.ndarray:
    """
    Per-trial features from epoched EEG.

    X: (trials, channels, samples)

    use_frequency_features=False (default, legacy):
      mean, std, variance, min, max, median per channel → 240 features (DEAP).

    use_frequency_features=True (Step 19):
      same 6 statistical features + delta/theta/alpha/beta/gamma Welch band powers
      per channel → 440 features (DEAP).
    """
    if X.ndim != 3:
        raise ValueError(f"Expected X with shape (trials, channels, samples), got {X.shape}")

    if use_frequency_features:
        return _extract_features_with_frequency(X, fs=fs)
    return _extract_features_legacy(X, fs=fs)
