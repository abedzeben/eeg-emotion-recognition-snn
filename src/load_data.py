from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Dataset:
    """
    Simple in-memory dataset container.

    X: shape (n_samples, n_channels, n_times) or (n_samples, n_features)
    y: shape (n_samples,)
    sfreq: sampling frequency in Hz (if time-series)
    ch_names: optional channel names
    """

    X: np.ndarray
    y: np.ndarray
    sfreq: Optional[float] = None
    ch_names: Optional[list[str]] = None
    meta: Optional[pd.DataFrame] = None


def load_raw_csv(
    path: str | Path,
    *,
    label_col: str = "label",
    subject_col: str | None = None,
) -> Dataset:
    """
    Load a flat CSV where each row is a sample and columns are features.

    Expected:
    - one column with labels (default: 'label')
    - remaining numeric columns are features
    """
    df = pd.read_csv(Path(path))
    if label_col not in df.columns:
        raise ValueError(f"label_col '{label_col}' not found in CSV columns: {list(df.columns)}")

    y = df[label_col].to_numpy()
    feature_df = df.drop(columns=[label_col] + ([subject_col] if subject_col in df.columns else []))
    X = feature_df.to_numpy(dtype=float)

    meta_cols = [c for c in [subject_col] if c and c in df.columns]
    meta = df[meta_cols].copy() if meta_cols else None
    return Dataset(X=X, y=y, meta=meta)


def load_synthetic_eeg(
    *,
    n_samples: int = 200,
    n_channels: int = 14,
    n_times: int = 256,
    n_classes: int = 3,
    sfreq: float = 128.0,
    seed: int = 7,
) -> Dataset:
    """
    Generate a small synthetic EEG-like dataset for smoke-testing the pipeline.
    """
    rng = np.random.default_rng(seed)
    X = rng.standard_normal(size=(n_samples, n_channels, n_times)).astype(np.float32)
    y = rng.integers(0, n_classes, size=(n_samples,), dtype=np.int64)
    ch_names = [f"Ch{i+1:02d}" for i in range(n_channels)]
    return Dataset(X=X, y=y, sfreq=sfreq, ch_names=ch_names)


def train_test_split_dataset(
    ds: Dataset,
    *,
    test_size: float = 0.2,
    seed: int = 7,
) -> Tuple[Dataset, Dataset]:
    rng = np.random.default_rng(seed)
    n = ds.X.shape[0]
    idx = np.arange(n)
    rng.shuffle(idx)
    split = int((1.0 - test_size) * n)
    tr_idx, te_idx = idx[:split], idx[split:]

    def _sub(ix: np.ndarray) -> Dataset:
        meta = ds.meta.iloc[ix].reset_index(drop=True) if ds.meta is not None else None
        return Dataset(
            X=ds.X[ix],
            y=ds.y[ix],
            sfreq=ds.sfreq,
            ch_names=ds.ch_names,
            meta=meta,
        )

    return _sub(tr_idx), _sub(te_idx)
