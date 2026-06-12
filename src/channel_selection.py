from __future__ import annotations

import numpy as np

# DEAP 40-channel order (32 EEG + 8 peripheral)
NUM_EEG_CHANNELS = 32
PERIPHERAL_CHANNEL_NAMES: list[str] = [
    "hEOG", "vEOG", "zEMG", "tEMG", "GSR", "Resp", "Pleth", "Temp",
]

DEAP_CHANNEL_NAMES: list[str] = [
    "Fp1", "AF3", "F3", "F7", "FC5", "FC1", "C3", "T7", "CP5", "CP1",
    "P3", "P7", "PO3", "O1", "Oz", "Pz", "Fp2", "AF4", "F4", "F8",
    "FC6", "FC2", "C4", "T8", "CP6", "CP2", "P4", "P8", "PO4", "O2",
    "Fz", "Cz",
    *PERIPHERAL_CHANNEL_NAMES,
]

EEG_CHANNEL_NAMES: list[str] = DEAP_CHANNEL_NAMES[:NUM_EEG_CHANNELS]

CHANNEL_SELECTION_MODES: dict[str, list[str]] = {
    "all": list(DEAP_CHANNEL_NAMES),
    "frontal": ["Fp1", "Fp2", "F3", "F4", "F7", "F8", "Fz"],
    "frontal_temporal": [
        "Fp1", "Fp2", "F3", "F4", "F7", "F8", "Fz",
        "T7", "T8", "FC5", "FC6",
    ],
}


def _resolve_channel_indices(channel_names: list[str]) -> list[int]:
    """Map channel names to DEAP tensor indices."""
    indices: list[int] = []
    for name in channel_names:
        try:
            indices.append(DEAP_CHANNEL_NAMES.index(name))
        except ValueError as exc:
            raise ValueError(f"Unknown DEAP channel: {name}") from exc
    return indices


def select_channels(
    X: np.ndarray,
    mode: str = "all",
    *,
    enabled: bool = True,
) -> tuple[np.ndarray, list[str]]:
    """
    Subset EEG trials along the channel axis.

    X: (trials, channels, samples)
    Returns: (X_selected, selected_channel_names)
    """
    if X.ndim != 3:
        raise ValueError(f"Expected X with shape (trials, channels, samples), got {X.shape}")

    if mode not in CHANNEL_SELECTION_MODES:
        valid = ", ".join(sorted(CHANNEL_SELECTION_MODES))
        raise ValueError(f"Unknown channel selection mode '{mode}'. Valid modes: {valid}")

    if not enabled:
        n_channels = X.shape[1]
        if n_channels == len(DEAP_CHANNEL_NAMES):
            return X, list(DEAP_CHANNEL_NAMES)
        return X, [f"Ch{i}" for i in range(n_channels)]

    channel_names = CHANNEL_SELECTION_MODES[mode]
    if mode == "all":
        n_channels = X.shape[1]
        if n_channels == len(DEAP_CHANNEL_NAMES):
            return X, list(DEAP_CHANNEL_NAMES)
        return X, [DEAP_CHANNEL_NAMES[i] if i < len(DEAP_CHANNEL_NAMES) else f"Ch{i}" for i in range(n_channels)]

    indices = _resolve_channel_indices(channel_names)
    return X[:, indices, :], list(channel_names)


def select_eeg_only_channels(X: np.ndarray) -> tuple[np.ndarray, list[str]]:
    """
    Step 33: keep only the 32 EEG channels (exclude peripheral sensors).

    X: (trials, channels, samples)
    Returns: (X_eeg, EEG_CHANNEL_NAMES)
    """
    if X.ndim != 3:
        raise ValueError(f"Expected X with shape (trials, channels, samples), got {X.shape}")
    if X.shape[1] < NUM_EEG_CHANNELS:
        raise ValueError(
            f"Expected at least {NUM_EEG_CHANNELS} channels for EEG-only selection, got {X.shape[1]}"
        )
    return X[:, :NUM_EEG_CHANNELS, :].astype(np.float32, copy=False), list(EEG_CHANNEL_NAMES)


def print_eeg_only_channel_info(n_channels: int) -> None:
    """Print Step 33 EEG-only channel summary."""
    print("\n=== EEG-only temporal SNN channels (Step 33) ===")
    print("SNN_USE_EEG_ONLY_CHANNELS: enabled")
    print("Number of EEG channels used:", n_channels)
    print("Excluded peripheral channels:", ", ".join(PERIPHERAL_CHANNEL_NAMES))


def print_channel_selection_info(
    mode: str,
    selected_channels: list[str],
    *,
    enabled: bool = True,
) -> None:
    """Print active channel selection configuration."""
    if not enabled:
        print("\nChannel selection: disabled (using all input channels)")
        return

    print("\n=== Channel selection ===")
    print(f"Channel selection mode: {mode}")
    print(f"Number of selected channels: {len(selected_channels)}")
    print(f"Selected channels: {', '.join(selected_channels)}")
