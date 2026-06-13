from __future__ import annotations

from typing import List, Literal, Optional, Tuple

TEMPORAL_FEATURE_TYPES = ("de", "log_psd", "de_log_psd")
TemporalFeatureType = Literal["de", "log_psd", "de_log_psd"]

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
        "features_per_channel": 6,
        "description": "mean, std, variance, min, max, median per channel",
    },
    "frequency": {
        "label": "Frequency",
        "expected_deap_size": 440,
        "features_per_channel": 11,
        "description": "6 statistical + 5 Welch band powers per channel",
    },
    "differential_entropy": {
        "label": "Differential Entropy",
        "expected_deap_size": 200,
        "features_per_channel": 5,
        "description": "5 band-pass DE values per channel (delta–gamma)",
    },
    "combined_stat_de": {
        "label": "Combined Statistical + Differential Entropy",
        "expected_deap_size": 440,
        "features_per_channel": 11,
        "description": "6 statistical + 5 DE values per channel",
    },
}


def get_expected_feature_size(feature_mode: str, n_channels: int) -> int:
    """Expected feature vector length for a given mode and channel count."""
    info = FEATURE_MODES[feature_mode]
    per_channel = info.get("features_per_channel")
    if per_channel is None:
        per_channel = info["expected_deap_size"] // 40
    return per_channel * n_channels


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


def get_temporal_features_per_window(
    n_channels: int,
    feature_type: TemporalFeatureType = "de",
    *,
    use_frontal_asymmetry: bool = False,
) -> int:
    """Expected feature count per temporal window for a channel count and feature type."""
    n_bands = len(FREQUENCY_BANDS)
    if feature_type == "de":
        base = n_channels * n_bands
    elif feature_type == "log_psd":
        base = n_channels * n_bands
    elif feature_type == "de_log_psd":
        base = n_channels * n_bands * 2
    else:
        raise ValueError(f"Unknown temporal feature type: {feature_type}")
    if use_frontal_asymmetry:
        base += FRONTAL_ASYMMETRY_FEATURES_PER_WINDOW
    return base


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


TEMPORAL_NUM_WINDOWS = 10
TEMPORAL_FEATURES_PER_WINDOW = 200  # 40 channels × 5 DE bands
FRONTAL_ASYMMETRY_FEATURES_PER_WINDOW = 15  # 3 pairs × 5 DE bands

# Left / right frontal pairs for valence-related asymmetry (right − left)
FRONTAL_ASYMMETRY_PAIRS: Tuple[Tuple[str, str], ...] = (
    ("F3", "F4"),
    ("F7", "F8"),
    ("Fp1", "Fp2"),
)

# Step 45: symmetric difference channels (left − right) for literature-style asymmetry SNN
SYMMETRIC_DIFFERENCE_PAIRS: Tuple[Tuple[str, str], ...] = (
    ("Fp1", "Fp2"),
    ("AF3", "AF4"),
    ("F3", "F4"),
    ("F7", "F8"),
    ("FC5", "FC6"),
    ("FC1", "FC2"),
    ("C3", "C4"),
    ("T7", "T8"),
    ("CP5", "CP6"),
    ("CP1", "CP2"),
    ("P3", "P4"),
    ("P7", "P8"),
    ("PO3", "PO4"),
    ("O1", "O2"),
)

# Theta, Alpha, Beta, Gamma, High Gamma (no delta — Step 45 literature bands)
ASYMMETRY_DE_BANDS: list[tuple[float, float]] = [
    (4, 8),     # Theta
    (8, 13),    # Alpha
    (13, 30),   # Beta
    (30, 45),   # Gamma
    (45, 63),   # High Gamma (below Nyquist for 128 Hz)
]

NUM_SYMMETRIC_DIFFERENCE_CHANNELS = len(SYMMETRIC_DIFFERENCE_PAIRS)
ASYMMETRY_FEATURES_PER_WINDOW = NUM_SYMMETRIC_DIFFERENCE_CHANNELS * len(ASYMMETRY_DE_BANDS)


def _trial_differential_entropy_bands(
    trial: np.ndarray,
    fs: float,
    bands: list[tuple[float, float]],
) -> np.ndarray:
    """DE per channel for custom frequency bands."""
    de_bands = []
    for lo, hi in bands:
        filtered = _filter_band(trial, lo, hi, fs)
        band_var = np.var(filtered, axis=-1)
        de_bands.append(_differential_entropy(band_var))
    return np.stack(de_bands, axis=-1)


def compute_symmetric_difference_eeg(X: np.ndarray) -> np.ndarray:
    """
    Step 45: left − right symmetric difference channels.

    X: (trials, channels, samples)
    Returns: (trials, 14, samples)
    """
    if X.ndim != 3:
        raise ValueError(f"Expected X with shape (trials, channels, samples), got {X.shape}")

    from src.channel_selection import DEAP_CHANNEL_NAMES

    n_trials, _, n_samples = X.shape
    out = np.zeros((n_trials, NUM_SYMMETRIC_DIFFERENCE_CHANNELS, n_samples), dtype=np.float32)
    for idx, (left_name, right_name) in enumerate(SYMMETRIC_DIFFERENCE_PAIRS):
        left_i = DEAP_CHANNEL_NAMES.index(left_name)
        right_i = DEAP_CHANNEL_NAMES.index(right_name)
        out[:, idx, :] = X[:, left_i, :] - X[:, right_i, :]
    return out


def extract_temporal_window_symmetric_de_features(
    X: np.ndarray,
    fs: float = 128.0,
    num_windows: int = TEMPORAL_NUM_WINDOWS,
) -> np.ndarray:
    """
    Step 45: windowed DE on 14 symmetric-difference channels.

    Bands: theta, alpha, beta, gamma, high gamma → 14 × 5 = 70 features/window.
    Returns: (trials, num_windows, 70)
    """
    X_sym = compute_symmetric_difference_eeg(X)
    if X_sym.ndim != 3:
        raise ValueError(f"Expected symmetric EEG (trials, 14, samples), got {X_sym.shape}")

    n_trials, n_channels, n_samples = X_sym.shape
    window_size = n_samples // num_windows
    if window_size < 1:
        raise ValueError(
            f"Trial length {n_samples} is too short for {num_windows} windows"
        )

    n_bands = len(ASYMMETRY_DE_BANDS)
    feats = np.zeros((n_trials, num_windows, n_channels * n_bands), dtype=np.float32)

    for trial_idx in range(n_trials):
        for window_idx in range(num_windows):
            start = window_idx * window_size
            end = start + window_size
            window = X_sym[trial_idx, :, start:end]
            de = _trial_differential_entropy_bands(window, fs, ASYMMETRY_DE_BANDS)
            feats[trial_idx, window_idx] = de.reshape(-1)

    return feats


def print_symmetric_difference_feature_info(
    X_baseline: np.ndarray,
    X_asymmetry: np.ndarray,
    *,
    num_windows: int = TEMPORAL_NUM_WINDOWS,
) -> None:
    """Print Step 45 baseline vs asymmetry temporal shapes."""
    print("\n=== Step 45 Symmetric Difference Features ===")
    print("Original temporal shape (40 ch × 5 DE bands):", X_baseline.shape)
    print("Asymmetry temporal shape (14 sym-diff ch × 5 bands):", X_asymmetry.shape)
    print("Symmetric difference pairs:", len(SYMMETRIC_DIFFERENCE_PAIRS))
    print("Bands:", [f"{lo}-{hi} Hz" for lo, hi in ASYMMETRY_DE_BANDS])
    print("Features per window (baseline):", X_baseline.shape[2])
    print("Features per window (asymmetry):", X_asymmetry.shape[2])
    print("Number of time steps:", num_windows)


def _deap_channel_index(channel_name: str) -> int:
    from src.channel_selection import DEAP_CHANNEL_NAMES

    return DEAP_CHANNEL_NAMES.index(channel_name)


def _frontal_asymmetry_from_de(de: np.ndarray) -> np.ndarray:
    """
    Compute frontal asymmetry from per-channel DE.

    de: (channels, n_bands)
    Returns: (15,) — right_DE − left_DE for each pair and band.
    """
    asym_parts: List[np.ndarray] = []
    for left_name, right_name in FRONTAL_ASYMMETRY_PAIRS:
        left_idx = _deap_channel_index(left_name)
        right_idx = _deap_channel_index(right_name)
        asym_parts.append(de[right_idx, :] - de[left_idx, :])
    return np.concatenate(asym_parts).astype(np.float32)


def extract_frontal_asymmetry_window_features(
    X: np.ndarray,
    fs: float = 128.0,
    num_windows: int = TEMPORAL_NUM_WINDOWS,
) -> np.ndarray:
    """
    Step 32: per-window frontal asymmetry from DE band features.

    X: (trials, channels, samples)
    Returns: (trials, num_windows, 15)
    """
    if X.ndim != 3:
        raise ValueError(f"Expected X with shape (trials, channels, samples), got {X.shape}")

    n_trials, _, n_samples = X.shape
    window_size = n_samples // num_windows
    if window_size < 1:
        raise ValueError(
            f"Trial length {n_samples} is too short for {num_windows} windows"
        )

    feats = np.zeros(
        (n_trials, num_windows, FRONTAL_ASYMMETRY_FEATURES_PER_WINDOW),
        dtype=np.float32,
    )
    for trial_idx in range(n_trials):
        for window_idx in range(num_windows):
            start = window_idx * window_size
            end = start + window_size
            window = X[trial_idx, :, start:end]
            de = _trial_differential_entropy(window, fs=fs)
            feats[trial_idx, window_idx] = _frontal_asymmetry_from_de(de)

    return feats


def extract_temporal_window_log_psd_features(
    X: np.ndarray,
    fs: float = 128.0,
    num_windows: int = TEMPORAL_NUM_WINDOWS,
) -> np.ndarray:
    """
    Step 33: per-window log PSD band features for temporal SNN input.

    Per window: n_channels × 5 bands (delta–gamma).
    Returns: (trials, num_windows, n_channels * 5)
    """
    if X.ndim != 3:
        raise ValueError(f"Expected X with shape (trials, channels, samples), got {X.shape}")

    n_trials, n_channels, n_samples = X.shape
    window_size = n_samples // num_windows
    if window_size < 1:
        raise ValueError(
            f"Trial length {n_samples} is too short for {num_windows} windows"
        )

    feats = np.zeros((n_trials, num_windows, n_channels * len(FREQUENCY_BANDS)), dtype=np.float32)
    for trial_idx in range(n_trials):
        for window_idx in range(num_windows):
            start = window_idx * window_size
            end = start + window_size
            window = X[trial_idx, :, start:end]
            log_psd = _trial_log_psd(window, fs=fs)
            feats[trial_idx, window_idx] = log_psd.reshape(-1)

    return feats


def extract_temporal_features_by_type(
    X: np.ndarray,
    feature_type: TemporalFeatureType = "de",
    fs: float = 128.0,
    num_windows: int = TEMPORAL_NUM_WINDOWS,
    *,
    use_frontal_asymmetry: bool = False,
) -> np.ndarray:
    """
    Step 33 temporal SNN features by type.

    de: windowed DE (n_channels × 5 per window)
    log_psd: windowed log Welch PSD (n_channels × 5 per window)
    de_log_psd: concatenate DE + log_psd (n_channels × 10 per window)
  """
    if feature_type == "de":
        base_feats = extract_temporal_window_de_features(X, fs=fs, num_windows=num_windows)
    elif feature_type == "log_psd":
        base_feats = extract_temporal_window_log_psd_features(X, fs=fs, num_windows=num_windows)
    elif feature_type == "de_log_psd":
        de_feats = extract_temporal_window_de_features(X, fs=fs, num_windows=num_windows)
        log_psd_feats = extract_temporal_window_log_psd_features(X, fs=fs, num_windows=num_windows)
        base_feats = np.concatenate([de_feats, log_psd_feats], axis=2).astype(np.float32)
    else:
        raise ValueError(f"Unknown temporal feature type: {feature_type}")

    if not use_frontal_asymmetry:
        return base_feats

    asym_feats = extract_frontal_asymmetry_window_features(X, fs=fs, num_windows=num_windows)
    return np.concatenate([base_feats, asym_feats], axis=2).astype(np.float32)


def extract_temporal_window_snn_features(
    X: np.ndarray,
    fs: float = 128.0,
    num_windows: int = TEMPORAL_NUM_WINDOWS,
    *,
    feature_type: TemporalFeatureType = "de",
    use_frontal_asymmetry: bool = False,
) -> np.ndarray:
    """
    Temporal SNN features with optional frontal asymmetry (Step 32).

    Default feature_type='de': (trials, num_windows, n_channels*5) or +15 asymmetry.
    """
    return extract_temporal_features_by_type(
        X,
        feature_type=feature_type,
        fs=fs,
        num_windows=num_windows,
        use_frontal_asymmetry=use_frontal_asymmetry,
    )


def print_temporal_feature_type_info(
    feature_type: TemporalFeatureType,
    X_temporal: np.ndarray,
    *,
    num_windows: int = TEMPORAL_NUM_WINDOWS,
) -> None:
    """Print Step 33 temporal feature type summary."""
    print("\n=== Temporal feature type (Step 33) ===")
    print("TEMPORAL_FEATURE_TYPE:", feature_type)
    print("Temporal feature shape:", X_temporal.shape)
    print("Features per window:", X_temporal.shape[2])
    print("Number of time steps:", num_windows)


def print_frontal_asymmetry_feature_info(
    X_temporal: np.ndarray,
    *,
    num_windows: int = TEMPORAL_NUM_WINDOWS,
) -> None:
    """Print Step 32 frontal asymmetry summary for temporal SNN input."""
    print("\n=== Frontal asymmetry features (Step 32) ===")
    print("Frontal asymmetry enabled")
    print("Number of asymmetry features per window:", FRONTAL_ASYMMETRY_FEATURES_PER_WINDOW)
    print("Temporal SNN feature shape:", X_temporal.shape)
    print("SNN input per time step:", X_temporal.shape[2])
    print("Number of time steps:", num_windows)


def extract_temporal_window_de_features(
    X: np.ndarray,
    fs: float = 128.0,
    num_windows: int = TEMPORAL_NUM_WINDOWS,
) -> np.ndarray:
    """
    Step 27: per-window Differential Entropy features for temporal SNN input.

    Splits each trial into fixed windows (window_size = trial_length // num_windows).
    Per window: 40 channels × 5 DE bands = 200 features.

    X: (trials, channels, samples) — use all 40 DEAP channels.
    Returns: (trials, num_windows, channels * 5)
    """
    if X.ndim != 3:
        raise ValueError(f"Expected X with shape (trials, channels, samples), got {X.shape}")

    n_trials, n_channels, n_samples = X.shape
    window_size = n_samples // num_windows
    if window_size < 1:
        raise ValueError(
            f"Trial length {n_samples} is too short for {num_windows} windows"
        )

    usable_samples = window_size * num_windows
    feats = np.zeros((n_trials, num_windows, n_channels * len(FREQUENCY_BANDS)), dtype=np.float32)

    for trial_idx in range(n_trials):
        for window_idx in range(num_windows):
            start = window_idx * window_size
            end = start + window_size
            window = X[trial_idx, :, start:end]
            de = _trial_differential_entropy(window, fs=fs)
            feats[trial_idx, window_idx] = de.reshape(-1)

    return feats


def print_temporal_snn_feature_info(
    X_temporal: np.ndarray,
    *,
    num_windows: int = TEMPORAL_NUM_WINDOWS,
) -> None:
    """Print Step 27 temporal SNN feature summary."""
    if X_temporal.ndim != 3:
        raise ValueError(
            f"Expected temporal features with shape (trials, windows, features), got {X_temporal.shape}"
        )
    features_per_step = X_temporal.shape[2]
    print("\n=== Temporal SNN features (Step 27) ===")
    print("Temporal SNN feature shape:", X_temporal.shape)
    print("SNN input per time step:", features_per_step)
    print("Number of time steps:", num_windows)


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
    bands: Optional[List[Tuple[float, float]]] = None,
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
    bands: Optional[List[Tuple[float, float]]] = None,
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


def _trial_log_psd(trial: np.ndarray, fs: float = 128.0) -> np.ndarray:
    """
    Log Welch PSD band power per channel.

    trial: (channels, samples)
    Returns: (channels, n_bands)
    """
    band_powers = _welch_band_powers(trial, fs=fs, bands=FREQUENCY_BANDS)
    eps = 1e-8
    return np.log(band_powers + eps).astype(np.float32)


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
