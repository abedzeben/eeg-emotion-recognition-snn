from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy import signal


@dataclass(frozen=True)
class PreprocessConfig:
    sfreq: float
    l_freq: Optional[float] = 1.0
    h_freq: Optional[float] = 40.0
    notch: Optional[float] = 50.0  # set 60.0 if needed
    standardize: bool = True


def _butter_bandpass(sfreq: float, l_freq: float, h_freq: float, order: int = 4):
    nyq = 0.5 * sfreq
    low = l_freq / nyq
    high = h_freq / nyq
    return signal.butter(order, [low, high], btype="bandpass")


def preprocess_eeg(X: np.ndarray, cfg: PreprocessConfig) -> np.ndarray:
    """
    Basic preprocessing for epoched EEG.

    X: (n_samples, n_channels, n_times)
    """
    if X.ndim != 3:
        raise ValueError(f"Expected X with shape (n_samples, n_channels, n_times), got {X.shape}")

    Xp = X.astype(np.float32, copy=True)

    if cfg.notch is not None:
        b, a = signal.iirnotch(w0=cfg.notch, Q=30.0, fs=cfg.sfreq)
        Xp = signal.filtfilt(b, a, Xp, axis=-1)

    if cfg.l_freq is not None and cfg.h_freq is not None:
        b, a = _butter_bandpass(cfg.sfreq, cfg.l_freq, cfg.h_freq, order=4)
        Xp = signal.filtfilt(b, a, Xp, axis=-1)

    if cfg.standardize:
        mean = Xp.mean(axis=-1, keepdims=True)
        std = Xp.std(axis=-1, keepdims=True) + 1e-6
        Xp = (Xp - mean) / std

    return Xp


def bandpass_filter(data: np.ndarray, low: float = 0.5, high: float = 50.0, fs: float = 128.0) -> np.ndarray:
    """
    Apply a Butterworth bandpass filter along the last axis.

    Works with DEAP-shaped arrays: (trials, channels, samples)
    """
    if data.ndim < 1:
        raise ValueError("data must be a numpy array with at least 1 dimension")

    nyq = 0.5 * fs
    low_norm = low / nyq
    high_norm = high / nyq
    if not (0.0 < low_norm < high_norm < 1.0):
        raise ValueError(f"Invalid band: low={low}, high={high}, fs={fs}")

    b, a = signal.butter(4, [low_norm, high_norm], btype="bandpass")
    return signal.lfilter(b, a, data, axis=-1).astype(np.float32, copy=False)


def normalize(data: np.ndarray) -> np.ndarray:
    """
    Normalize each trial/channel signal: (x - mean) / (std + eps) along last axis.
    """
    eps = 1e-8
    mean = np.mean(data, axis=-1, keepdims=True)
    std = np.std(data, axis=-1, keepdims=True)
    return ((data - mean) / (std + eps)).astype(np.float32, copy=False)
