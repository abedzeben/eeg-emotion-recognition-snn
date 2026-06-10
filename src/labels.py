from __future__ import annotations

import numpy as np

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
    label_map: dict[int, str] | None = None,
    *,
    num_classes: int | None = None,
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
