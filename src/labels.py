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


def create_multi_emotion_labels(y: np.ndarray) -> np.ndarray:
    """
    Map DEAP Valence-Arousal scores to 4 emotion classes.

    Input: y shape (n_samples, 4+) with Valence=y[:,0], Arousal=y[:,1]
    """
    if y.ndim != 2 or y.shape[1] < 2:
        raise ValueError(f"Expected y with shape (n_samples, >=2), got {y.shape}")

    valence = y[:, 0]
    arousal = y[:, 1]

    # 0 = Calm / Relaxed:   Valence > median(valence) and Arousal <= median(arousal)
    # 1 = Happy / Excited:  Valence > median(valence) and Arousal > median(arousal)
    # 2 = Sad / Low:        Valence <= median(valence) and Arousal <= median(arousal)
    # 3 = Angry / Stressed: Valence <= median(valence) and Arousal > median(arousal)
    labels = np.empty((y.shape[0],), dtype=np.int64)

    v_threshold = np.median(valence)
    a_threshold = np.median(arousal)

    print("\nMedian thresholds:")
    print("Valence threshold:", v_threshold)
    print("Arousal threshold:", a_threshold)

    v_pos = valence > v_threshold
    a_pos = arousal > a_threshold

    labels[v_pos & ~a_pos] = 0
    labels[v_pos & a_pos] = 1
    labels[~v_pos & ~a_pos] = 2
    labels[~v_pos & a_pos] = 3
    print("\nDEBUG MULTI LABELS")

    print("Valence stats:")
    print("min:", valence.min())
    print("max:", valence.max())
    print("mean:", valence.mean())

    print("Arousal stats:")
    print("min:", arousal.min())
    print("max:", arousal.max())
    print("mean:", arousal.mean())

    print("HVHA:", ((valence > v_threshold) & (arousal > a_threshold)).sum())
    print("HVLA:", ((valence > v_threshold) & (arousal <= a_threshold)).sum())
    print("LVLA:", ((valence <= v_threshold) & (arousal <= a_threshold)).sum())
    print("LVHA:", ((valence <= v_threshold) & (arousal > a_threshold)).sum())

    return labels


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
