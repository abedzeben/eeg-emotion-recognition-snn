from __future__ import annotations

import numpy as np
from typing import Any, Dict, List, Optional, Tuple

EMOTION_LABELS = {
    0: "Calm / Relaxed",
    1: "Happy / Excited",
    2: "Sad / Low",
    3: "Angry / Stressed",
}

BINARY_LABELS = {
    0: "Calm",
    1: "Excited",
}

VALENCE_BINARY_LABELS = {
    0: "Low Valence (<=4.5)",
    1: "High Valence (>4.5)",
}

AROUSAL_BINARY_LABELS = {
    0: "Low Arousal (<=4.5)",
    1: "High Arousal (>4.5)",
}

BINARY_VALIDATION_THRESHOLD = 4.5

LABEL_STRATEGIES: tuple[str, ...] = (
    "median",
    "mean",
    "fixed_5",
    "quantile_60",
    "quantile_40",
)


def _compute_thresholds(
    valence: np.ndarray,
    arousal: np.ndarray,
    strategy: str,
) -> tuple[float, float]:
    """Compute Valence and Arousal thresholds for a given strategy."""
    if strategy == "median":
        return float(np.median(valence)), float(np.median(arousal))
    if strategy == "mean":
        return float(np.mean(valence)), float(np.mean(arousal))
    if strategy == "fixed_5":
        return 5.0, 5.0
    if strategy == "quantile_60":
        return float(np.quantile(valence, 0.6)), float(np.quantile(arousal, 0.6))
    if strategy == "quantile_40":
        return float(np.quantile(valence, 0.4)), float(np.quantile(arousal, 0.4))
    raise ValueError(
        f"Unknown strategy '{strategy}'. Choose from: {', '.join(LABEL_STRATEGIES)}"
    )


def _apply_quadrant_labels(
    valence: np.ndarray,
    arousal: np.ndarray,
    v_threshold: float,
    a_threshold: float,
) -> np.ndarray:
    """
    Map Valence-Arousal scores to 4 classes using given thresholds.

    0 = Calm / Relaxed:   V > t_v, A <= t_a
    1 = Happy / Excited:  V > t_v, A > t_a
    2 = Sad / Low:        V <= t_v, A <= t_a
    3 = Angry / Stressed: V <= t_v, A > t_a
    """
    labels = np.empty((valence.shape[0],), dtype=np.int64)
    v_pos = valence > v_threshold
    a_pos = arousal > a_threshold

    labels[v_pos & ~a_pos] = 0
    labels[v_pos & a_pos] = 1
    labels[~v_pos & ~a_pos] = 2
    labels[~v_pos & a_pos] = 3
    return labels


def get_empty_classes(y_multi: np.ndarray, num_classes: int = 4) -> list[int]:
    """Return class indices with zero samples."""
    counts = np.bincount(y_multi.astype(int), minlength=num_classes)
    return [cls for cls in range(num_classes) if counts[cls] == 0]


def create_valence_binary_labels(
    y: np.ndarray,
    threshold: float = BINARY_VALIDATION_THRESHOLD,
) -> np.ndarray:
    """Step 43: 0 if Valence <= threshold, 1 if Valence > threshold."""
    if y.ndim != 2 or y.shape[1] < 1:
        raise ValueError(f"Expected y with shape (n_samples, >=1), got {y.shape}")
    return (y[:, 0] > threshold).astype(np.int64)


def create_arousal_binary_labels(
    y: np.ndarray,
    threshold: float = BINARY_VALIDATION_THRESHOLD,
) -> np.ndarray:
    """Step 43: 0 if Arousal <= threshold, 1 if Arousal > threshold."""
    if y.ndim != 2 or y.shape[1] < 2:
        raise ValueError(f"Expected y with shape (n_samples, >=2), got {y.shape}")
    return (y[:, 1] > threshold).astype(np.int64)


def print_binary_class_distribution(
    y_binary: np.ndarray,
    label_map: Dict[int, str],
    *,
    title: str = "Class distribution",
) -> Dict[str, Any]:
    """Print and return per-class counts and percentages."""
    y_binary = np.asarray(y_binary).astype(int)
    n = int(y_binary.shape[0])
    counts = np.bincount(y_binary, minlength=2)
    print(f"\n{title}")
    distribution: Dict[str, Any] = {}
    for cls in range(2):
        name = label_map.get(cls, str(cls))
        count = int(counts[cls])
        pct = (count / n * 100.0) if n else 0.0
        print(f"  {name}: {count} ({pct:.2f}%)")
        distribution[name] = {"count": count, "percentage": round(pct, 4)}
    print(f"  Total: {n}")
    distribution["total"] = n
    return distribution


def create_multi_emotion_labels(
    y: np.ndarray,
    strategy: str = "median",
    *,
    verbose: bool = False,
) -> np.ndarray:
    """
    Map DEAP Valence-Arousal scores to 4 emotion classes.

    Input: y shape (n_samples, 4+) with Valence=y[:,0], Arousal=y[:,1]
    """
    if y.ndim != 2 or y.shape[1] < 2:
        raise ValueError(f"Expected y with shape (n_samples, >=2), got {y.shape}")
    if strategy not in LABEL_STRATEGIES:
        raise ValueError(
            f"Unknown strategy '{strategy}'. Choose from: {', '.join(LABEL_STRATEGIES)}"
        )

    valence = y[:, 0]
    arousal = y[:, 1]
    v_threshold, a_threshold = _compute_thresholds(valence, arousal, strategy)
    labels = _apply_quadrant_labels(valence, arousal, v_threshold, a_threshold)

    if verbose:
        print(f"Selected label strategy: {strategy}")
        print("Valence threshold:", v_threshold)
        print("Arousal threshold:", a_threshold)

    return labels


def compare_label_strategies(y: np.ndarray, *, num_classes: int = 4) -> None:
    """Print thresholds and class distribution for each supported strategy."""
    print("\n=== Label strategy comparison ===")
    for strategy in LABEL_STRATEGIES:
        valence = y[:, 0]
        arousal = y[:, 1]
        v_threshold, a_threshold = _compute_thresholds(valence, arousal, strategy)
        labels = _apply_quadrant_labels(valence, arousal, v_threshold, a_threshold)

        print(f"\nStrategy: {strategy}")
        print("Valence threshold:", v_threshold)
        print("Arousal threshold:", a_threshold)
        print("Class distribution:")
        print_class_distribution(labels, EMOTION_LABELS, num_classes=num_classes)

        empty = get_empty_classes(labels, num_classes=num_classes)
        if empty:
            empty_names = [EMOTION_LABELS[c] for c in empty]
            print(f"WARNING: empty classes for strategy '{strategy}': {empty} ({empty_names})")


def print_class_distribution(
    y: np.ndarray,
    label_map: Optional[Dict[int, str]] = None,
    *,
    num_classes: Optional[int] = None,
) -> None:
    """Print per-class sample counts."""
    if label_map is None:
        label_map = EMOTION_LABELS
    y = np.asarray(y)
    if num_classes is None:
        num_classes = int(y.max()) + 1 if y.size else 0

    counts = np.bincount(y.astype(int), minlength=num_classes) if y.size else np.zeros((num_classes,))
    for cls in range(num_classes):
        name = label_map.get(int(cls), str(cls))
        print(f"  Class {cls} ({name}): {int(counts[cls])}")


def create_clear_multi_emotion_labels(
    y: np.ndarray,
    *,
    low_threshold: float = 4.0,
    high_threshold: float = 6.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Step 30: keep only clear Valence-Arousal samples and assign fixed-quadrant labels.

    Keep when (V <= low or V >= high) AND (A <= low or A >= high).

    Class 0 Calm / Relaxed:   V >= high, A <= low
    Class 1 Happy / Excited:  V >= high, A >= high
    Class 2 Sad / Low:        V <= low, A <= low
    Class 3 Angry / Stressed: V <= low, A >= high

    Returns:
        keep_mask: bool array (n_samples,)
        labels: int64 array; discarded samples are -1
    """
    if y.ndim != 2 or y.shape[1] < 2:
        raise ValueError(f"Expected y with shape (n_samples, >=2), got {y.shape}")

    valence = y[:, 0]
    arousal = y[:, 1]

    v_clear = (valence <= low_threshold) | (valence >= high_threshold)
    a_clear = (arousal <= low_threshold) | (arousal >= high_threshold)
    keep_mask = v_clear & a_clear

    labels = np.full(valence.shape[0], -1, dtype=np.int64)
    v_high = valence >= high_threshold
    v_low = valence <= low_threshold
    a_high = arousal >= high_threshold
    a_low = arousal <= low_threshold

    labels[keep_mask & v_high & a_low] = 0
    labels[keep_mask & v_high & a_high] = 1
    labels[keep_mask & v_low & a_low] = 2
    labels[keep_mask & v_low & a_high] = 3

    return keep_mask, labels


def get_small_classes(
    y_multi: np.ndarray,
    *,
    num_classes: int = 4,
    min_samples: int = 10,
) -> list[int]:
    """Return class indices with fewer than min_samples."""
    counts = np.bincount(y_multi.astype(int), minlength=num_classes)
    return [cls for cls in range(num_classes) if counts[cls] < min_samples]


def print_ambiguous_sample_filter_summary(
    keep_mask: np.ndarray,
    labels: np.ndarray,
    *,
    low_threshold: float,
    high_threshold: float,
    min_class_samples: int = 10,
) -> None:
    """Print Step 30 ambiguous-sample filtering statistics."""
    original_n = int(keep_mask.shape[0])
    remaining_n = int(np.sum(keep_mask))
    removed_n = original_n - remaining_n
    y_kept = labels[keep_mask]

    print("\n=== Ambiguous sample filtering (Step 30) ===")
    print(f"LOW_THRESHOLD: {low_threshold}")
    print(f"HIGH_THRESHOLD: {high_threshold}")
    print("Original number of samples:", original_n)
    print("Remaining number of samples:", remaining_n)
    print("Removed number of samples:", removed_n)
    print("Class distribution after filtering:")
    print_class_distribution(y_kept, EMOTION_LABELS, num_classes=4)

    small_classes = get_small_classes(y_kept, min_samples=min_class_samples)
    if small_classes:
        small_names = [EMOTION_LABELS[c] for c in small_classes]
        print(
            f"WARNING: classes with fewer than {min_class_samples} samples: "
            f"{small_classes} ({small_names})"
        )


def subset_arrays_by_mask(mask: np.ndarray, *arrays: np.ndarray) -> tuple[np.ndarray, ...]:
    """Return arrays indexed by a boolean keep mask."""
    return tuple(arr[mask] for arr in arrays)
