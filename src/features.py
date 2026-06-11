from __future__ import annotations

import numpy as np
from scipy import signal
from sklearn.feature_selection import VarianceThreshold

# Step 19 frequency bands (Welch PSD average power per band)
FREQUENCY_BANDS: list[tuple[float, float]] = [
    (0.5, 4),   # Delta
    (4, 8),     # Theta
    (8, 13),    # Alpha
    (13, 30),   # Beta
    (30, 45),   # Gamma
]


FEATURE_MODES = {
    "statistical": {
        "label": "Statistical",
        "expected_deap_size": 240,
        "description": "mean, std, variance, min, max, median per channel",
    },
    "frequency": {
        "label": "Frequency",
        "expected_deap_size": 440,
        "description": "6 statistical + 5 Welch band powers per channel",
    },
    "differential_entropy": {
        "label": "Differential Entropy",
        "expected_deap_size": 200,
        "description": "5 band-pass DE values per channel (delta–gamma)",
    },
    "combined_stat_de": {
        "label": "Combined Statistical + Differential Entropy",
        "expected_deap_size": 440,
        "description": "6 statistical + 5 DE values per channel",
    },
}


def get_feature_mode_name(
    *,
    use_frequency_features: bool = False,
    use_differential_entropy: bool = False,
    use_combined_stat_de: bool = False,
) -> str:
    """Return the active feature mode key."""
    if use_combined_stat_de:
        return "combined_stat_de"
    if use_differential_entropy:
        return "differential_entropy"
    if use_frequency_features:
        return "frequency"
    return "statistical"


def print_feature_mode_comparison(
    *,
    use_frequency_features: bool = False,
    use_differential_entropy: bool = False,
    use_combined_stat_de: bool = False,
) -> None:
    """Print available feature modes and highlight the active one."""
    active = get_feature_mode_name(
        use_frequency_features=use_frequency_features,
        use_differential_entropy=use_differential_entropy,
        use_combined_stat_de=use_combined_stat_de,
    )
    print("\n=== Feature mode comparison ===")
    for key, info in FEATURE_MODES.items():
        marker = " (active)" if key == active else ""
        if key == "combined_stat_de":
            name = info["label"]
        else:
            name = f"{info['label']} Features"
        print(
            f"  {name}{marker}: "
            f"{info['expected_deap_size']} features — {info['description']}"
        )


def _butter_bandpass_coeffs(low: float, high: float, fs: float, order: int = 4):
    nyq = 0.5 * fs
    return signal.butter(order, [low / nyq, high / nyq], btype="bandpass")


def _filter_band(data: np.ndarray, low: float, high: float, fs: float) -> np.ndarray:
    """Band-pass filter along the last axis."""
    b, a = _butter_bandpass_coeffs(low, high, fs)
    return signal.filtfilt(b, a, data, axis=-1)


def _differential_entropy(variance: np.ndarray) -> np.ndarray:
    """DE = 0.5 * log(2 * pi * e * variance)."""
    eps = 1e-8
    return (0.5 * np.log(2.0 * np.pi * np.e * (variance + eps))).astype(np.float32)


def _trial_differential_entropy(trial: np.ndarray, fs: float = 128.0) -> np.ndarray:
    """
    Compute DE per channel per band for one trial.

    trial: (channels, samples)
    Returns: (channels, n_bands)
    """
    de_bands = []
    for lo, hi in FREQUENCY_BANDS:
        filtered = _filter_band(trial, lo, hi, fs)
        band_var = np.var(filtered, axis=-1)
        de_bands.append(_differential_entropy(band_var))
    return np.stack(de_bands, axis=-1)


def _extract_differential_entropy_features(X: np.ndarray, fs: float = 128.0) -> np.ndarray:
    """
    Step 23: Differential Entropy features per channel per band.

    Per channel: 5 DE values (delta, theta, alpha, beta, gamma).
    For DEAP (40 channels): (num_trials, 200).
    """
    n_trials = X.shape[0]
    feats = np.stack(
        [_trial_differential_entropy(X[i], fs=fs) for i in range(n_trials)],
        axis=0,
    )
    return feats.reshape(n_trials, -1).astype(np.float32)


def _extract_combined_stat_de_features(X: np.ndarray, fs: float = 128.0) -> np.ndarray:
    """
    Step 24: statistical features + Differential Entropy features.

    Per channel: 6 statistical + 5 DE values.
    For DEAP (40 channels): (num_trials, 440) = 240 + 200.
    """
    stat_feats = _extract_features_legacy(X, fs=fs)
    de_feats = _extract_differential_entropy_features(X, fs=fs)
    return np.concatenate([stat_feats, de_feats], axis=1).astype(np.float32)


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
    use_differential_entropy: bool = False,
    use_combined_stat_de: bool = False,
) -> np.ndarray:
    """
    Per-trial features from epoched EEG.

    X: (trials, channels, samples)

    Priority: combined_stat_de > differential_entropy > frequency > statistical.

    use_combined_stat_de=True (Step 24):
      6 statistical + 5 DE per channel → 440 features (DEAP).

    use_differential_entropy=True (Step 23):
      5 DE values per channel → 200 features (DEAP).

    use_frequency_features=True (Step 19):
      6 statistical + 5 Welch band powers per channel → 440 features (DEAP).

    default (statistical):
      mean, std, variance, min, max, median per channel → 240 features (DEAP).
    """
    if X.ndim != 3:
        raise ValueError(f"Expected X with shape (trials, channels, samples), got {X.shape}")

    if use_combined_stat_de:
        return _extract_combined_stat_de_features(X, fs=fs)
    if use_differential_entropy:
        return _extract_differential_entropy_features(X, fs=fs)
    if use_frequency_features:
        return _extract_features_with_frequency(X, fs=fs)
    return _extract_features_legacy(X, fs=fs)


def remove_constant_features(
    X: np.ndarray,
    threshold: float = 0.0,
) -> tuple[np.ndarray, int]:
    """
    Remove constant or near-constant features using VarianceThreshold.

    Features with variance <= threshold are dropped.

    Returns:
        X_clean, n_removed
    """
    if X.ndim != 2:
        raise ValueError(f"Expected X with shape (n_samples, n_features), got {X.shape}")

    selector = VarianceThreshold(threshold=threshold)
    X_clean = selector.fit_transform(X)
    n_removed = X.shape[1] - X_clean.shape[1]
    return X_clean.astype(np.float32, copy=False), n_removed
