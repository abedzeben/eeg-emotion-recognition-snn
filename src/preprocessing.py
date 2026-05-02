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
