from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from src.deap_cnn_snn import DEAP_TEMPORAL_BASELINE
from src.features import TEMPORAL_NUM_WINDOWS, extract_temporal_window_de_features
from src.labels import EMOTION_LABELS, create_multi_emotion_labels, get_empty_classes, print_class_distribution
from src.snn_model import BEST_TEMPORAL_SNN_CONFIG, _scale_temporal_features, _train_single_snn

N_SPLITS = 5
CV_RANDOM_STATE = 42
MIN_CLASS_SAMPLES = 2
TRIALS_PER_SUBJECT = 40
METRICS_JSON = Path("results/metrics/deap_subject_dependent_snn_results.json")
METRICS_CSV = Path("results/metrics/deap_subject_dependent_snn_results.csv")
FULL_RUN_RECOMMEND_ACC = 0.70
FULL_RUN_RECOMMEND_MACRO_F1 = 0.65


def _subject_has_valid_classes(y: np.ndarray, *, min_samples: int = MIN_CLASS_SAMPLES) -> Tuple[bool, List[int]]:
    counts = np.bincount(y.astype(int), minlength=4)
    small = [cls for cls in range(4) if counts[cls] < min_samples]
    return len(small) == 0, small


def _train_evaluate_fold(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> Dict[str, float]:
    cfg = BEST_TEMPORAL_SNN_CONFIG
    scaler = StandardScaler()
    X_train_s, X_test_s = _scale_temporal_features(X_train, X_test, scaler)

    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = 4

    _model, _y_pred, metrics = _train_single_snn(
        X_train_s,
        y_train,
        X_test_s,
        y_test,
        hidden_size=cfg["hidden_size"],
        second_hidden_size=cfg["second_hidden_size"],
        beta=cfg["beta"],
        dropout=cfg["dropout"],
        num_steps=X_train_s.shape[1],
        learning_rate=cfg["learning_rate"],
        epochs=cfg["epochs"],
        class_weight_mode=cfg["class_weight"],
        device=device,
        num_classes=num_classes,
        verbose=False,
        temporal=True,
        temporal_spike_encoding=False,
    )
    return {
        "accuracy": float(metrics["accuracy"]),
        "macro_f1": float(metrics["macro_f1"]),
    }


def _evaluate_subject(
    subject_id: int,
    X_subject: np.ndarray,
    y_ratings_subject: np.ndarray,
    *,
    label_strategy: str,
) -> Optional[Dict[str, Any]]:
    print(f"\n{'=' * 60}")
    print(f"Subject {subject_id}")
    print("=" * 60)

    y_multi = create_multi_emotion_labels(
        y_ratings_subject,
        strategy=label_strategy,
        verbose=False,
    )

    print("Class distribution:")
    print_class_distribution(y_multi, EMOTION_LABELS, num_classes=4)

    empty = get_empty_classes(y_multi, num_classes=4)
    if empty:
        empty_names = [EMOTION_LABELS[c] for c in empty]
        print(
            f"WARNING: empty classes for subject {subject_id}: "
            f"{empty} ({empty_names}) — skipping subject"
        )
        return None

    valid, small_classes = _subject_has_valid_classes(y_multi)
    if not valid:
        small_names = [EMOTION_LABELS[c] for c in small_classes]
        print(
            f"WARNING: classes with fewer than {MIN_CLASS_SAMPLES} samples "
            f"for subject {subject_id}: {small_classes} ({small_names}) — skipping subject"
        )
        return None

    counts = np.bincount(y_multi.astype(int), minlength=4)
    class_distribution = {
        EMOTION_LABELS[cls]: {"count": int(counts[cls]), "percentage": round(counts[cls] / len(y_multi) * 100, 4)}
        for cls in range(4)
    }

    skf = StratifiedKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=CV_RANDOM_STATE,
    )

    fold_results: List[Dict[str, Any]] = []
    fold_accuracies: List[float] = []
    fold_macro_f1: List[float] = []

    try:
        for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X_subject, y_multi)):
            X_train, X_test = X_subject[train_idx], X_subject[test_idx]
            y_train, y_test = y_multi[train_idx], y_multi[test_idx]

            metrics = _train_evaluate_fold(X_train, y_train, X_test, y_test)
            fold_results.append(
                {
                    "fold": fold_idx + 1,
                    "train_samples": int(len(train_idx)),
                    "test_samples": int(len(test_idx)),
                    "accuracy": metrics["accuracy"],
                    "macro_f1": metrics["macro_f1"],
                }
            )
            fold_accuracies.append(metrics["accuracy"])
            fold_macro_f1.append(metrics["macro_f1"])
            print(
                f"  Fold {fold_idx + 1}: accuracy={metrics['accuracy']:.4f}, "
                f"macro_f1={metrics['macro_f1']:.4f} "
                f"(train={len(train_idx)}, test={len(test_idx)})"
            )
    except ValueError as exc:
        print(f"WARNING: StratifiedKFold failed for subject {subject_id}: {exc} — skipping subject")
        return None

    mean_acc = float(np.mean(fold_accuracies))
    mean_f1 = float(np.mean(fold_macro_f1))

    print(f"Fold accuracies: {[round(a, 4) for a in fold_accuracies]}")
    print(f"Fold Macro F1:   {[round(f, 4) for f in fold_macro_f1]}")
    print(f"Mean accuracy:   {mean_acc:.4f}")
    print(f"Mean Macro F1:   {mean_f1:.4f}")

    return {
        "subject_id": subject_id,
        "trials": int(X_subject.shape[0]),
        "class_distribution": class_distribution,
        "folds": fold_results,
        "fold_accuracies": fold_accuracies,
        "fold_macro_f1": fold_macro_f1,
        "mean_accuracy": mean_acc,
        "mean_macro_f1": mean_f1,
    }


def _print_summary(
    subject_results: List[Dict[str, Any]],
    skipped_subjects: List[Dict[str, Any]],
    *,
    fast_mode: bool,
    subject_ids_used: List[int],
) -> None:
    ref = DEAP_TEMPORAL_BASELINE
    print("\n" + "=" * 60)
    print("=== DEAP Subject-Dependent SNN Summary ===")
    print("=" * 60)
    print("Number of subjects evaluated:", len(subject_results))
    print("Number of subjects skipped:", len(skipped_subjects))

    if not subject_results:
        print("No subjects evaluated.")
        return

    mean_accs = [r["mean_accuracy"] for r in subject_results]
    mean_f1s = [r["mean_macro_f1"] for r in subject_results]

    overall_mean_acc = float(np.mean(mean_accs))
    overall_std_acc = float(np.std(mean_accs))
    overall_mean_f1 = float(np.mean(mean_f1s))
    overall_std_f1 = float(np.std(mean_f1s))

    print(f"Overall mean accuracy:   {overall_mean_acc:.4f}")
    print(f"Overall std accuracy:  {overall_std_acc:.4f}")
    print(f"Overall mean Macro F1: {overall_mean_f1:.4f}")
    print(f"Overall std Macro F1:  {overall_std_f1:.4f}")
    print(f"Best subject accuracy:  {max(mean_accs):.4f} (subject {subject_results[int(np.argmax(mean_accs))]['subject_id']})")
    print(f"Worst subject accuracy: {min(mean_accs):.4f} (subject {subject_results[int(np.argmin(mean_accs))]['subject_id']})")

    print("\nComparison vs pooled DEAP Temporal SNN (Step 29 reference):")
    print(f"  Pooled accuracy:  {ref['accuracy']:.4f}")
    print(f"  Pooled Macro F1: {ref['macro_f1']:.4f}")
    print(f"  Subject-dependent mean accuracy delta:  {overall_mean_acc - ref['accuracy']:+.4f}")
    print(f"  Subject-dependent mean Macro F1 delta: {overall_mean_f1 - ref['macro_f1']:+.4f}")

    if fast_mode:
        print("\n--- Fast-run recommendation ---")
        print(f"Fast-run mean accuracy:   {overall_mean_acc:.4f}")
        print(f"Fast-run mean Macro F1:   {overall_mean_f1:.4f}")
        if overall_mean_acc > FULL_RUN_RECOMMEND_ACC or overall_mean_f1 > FULL_RUN_RECOMMEND_MACRO_F1:
            print(
                f"RECOMMENDATION: Fast-run exceeds thresholds "
                f"(accuracy > {FULL_RUN_RECOMMEND_ACC:.0%} or Macro F1 > {FULL_RUN_RECOMMEND_MACRO_F1:.2f}). "
                "Run full 32-subject experiment: SUBJECT_DEPENDENT_FAST_MODE = False"
            )
        else:
            print(
                f"RECOMMENDATION: Fast-run did NOT exceed thresholds "
                f"(accuracy > {FULL_RUN_RECOMMEND_ACC:.0%} or Macro F1 > {FULL_RUN_RECOMMEND_MACRO_F1:.2f}). "
                "Investigate before launching full 32-subject run."
            )
        print(f"Subjects used in fast run: {subject_ids_used}")


def _save_results(payload: Dict[str, Any]) -> None:
    METRICS_JSON.parent.mkdir(parents=True, exist_ok=True)

    with open(METRICS_JSON, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    rows = []
    for entry in payload.get("subject_results", []):
        rows.append(
            {
                "subject_id": entry["subject_id"],
                "trials": entry["trials"],
                "mean_accuracy": entry["mean_accuracy"],
                "mean_macro_f1": entry["mean_macro_f1"],
                "fold_accuracies": json.dumps(entry["fold_accuracies"]),
                "fold_macro_f1": json.dumps(entry["fold_macro_f1"]),
            }
        )

    with open(METRICS_CSV, "w", encoding="utf-8", newline="") as f:
        if rows:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    print(f"\nResults saved:")
    print(f"  {METRICS_JSON}")
    print(f"  {METRICS_CSV}")


def run_deap_subject_dependent_snn(
    folder: str = "data/raw",
    *,
    label_strategy: str = "mean",
    fast_mode: bool = True,
    max_subjects: int = 5,
    trials_per_subject: int = TRIALS_PER_SUBJECT,
) -> Dict[str, Any]:
    """
    Step 44: subject-dependent 4-class Temporal SNN with per-subject stratified 5-fold CV.
    """
    from src.load_data import load_all_deap_files
    from src.preprocessing import bandpass_filter

    print("\n" + "=" * 60)
    print("DEAP Subject-Dependent 4-Class Temporal SNN (Step 44)")
    print("=" * 60)
    print("SUBJECT_DEPENDENT_FAST_MODE:", fast_mode)
    print("MAX_SUBJECTS_FOR_SUBJECT_DEPENDENT:", max_subjects if fast_mode else "all")
    print("Label strategy:", label_strategy)
    print("Model config:", BEST_TEMPORAL_SNN_CONFIG)
    print(f"Cross-validation: StratifiedKFold n_splits={N_SPLITS}, shuffle=True, random_state={CV_RANDOM_STATE}")
    print("Reference pooled baseline:", DEAP_TEMPORAL_BASELINE)

    load_max = max_subjects if fast_mode else None
    X, y_ratings = load_all_deap_files(folder, max_subjects=load_max)
    X_filtered = bandpass_filter(X)

    n_subjects_loaded = X_filtered.shape[0] // trials_per_subject
    if fast_mode:
        subject_ids = list(range(n_subjects_loaded))
        print(f"\nUsing subjects: {subject_ids}")
        print(f"Total trainings: {len(subject_ids)} subjects x {N_SPLITS} folds = {len(subject_ids) * N_SPLITS} SNN trainings")
    else:
        subject_ids = list(range(n_subjects_loaded))
        print(f"\nSUBJECT_DEPENDENT_FAST_MODE = False")
        print(f"Using all {n_subjects_loaded} subjects")
        print(f"Total trainings: {n_subjects_loaded} subjects x {N_SPLITS} folds = {n_subjects_loaded * N_SPLITS} SNN trainings")

    X_temporal = extract_temporal_window_de_features(
        X_filtered,
        num_windows=TEMPORAL_NUM_WINDOWS,
    )
    print("\nTemporal feature shape:", X_temporal.shape)

    subject_results: List[Dict[str, Any]] = []
    skipped_subjects: List[Dict[str, Any]] = []

    for subject_id in subject_ids:
        start = subject_id * trials_per_subject
        end = start + trials_per_subject
        X_subj = X_temporal[start:end]
        y_subj = y_ratings[start:end]

        result = _evaluate_subject(
            subject_id,
            X_subj,
            y_subj,
            label_strategy=label_strategy,
        )
        if result is None:
            skipped_subjects.append({"subject_id": subject_id, "reason": "invalid_class_distribution_or_cv_failure"})
        else:
            subject_results.append(result)

    summary: Dict[str, Any] = {}
    if subject_results:
        mean_accs = [r["mean_accuracy"] for r in subject_results]
        mean_f1s = [r["mean_macro_f1"] for r in subject_results]
        summary = {
            "subjects_evaluated": len(subject_results),
            "subjects_skipped": len(skipped_subjects),
            "overall_mean_accuracy": float(np.mean(mean_accs)),
            "overall_std_accuracy": float(np.std(mean_accs)),
            "overall_mean_macro_f1": float(np.mean(mean_f1s)),
            "overall_std_macro_f1": float(np.std(mean_f1s)),
            "best_subject_accuracy": float(max(mean_accs)),
            "best_subject_id": int(subject_results[int(np.argmax(mean_accs))]["subject_id"]),
            "worst_subject_accuracy": float(min(mean_accs)),
            "worst_subject_id": int(subject_results[int(np.argmin(mean_accs))]["subject_id"]),
        }

    _print_summary(
        subject_results,
        skipped_subjects,
        fast_mode=fast_mode,
        subject_ids_used=subject_ids,
    )

    payload: Dict[str, Any] = {
        "study": "Step 44 DEAP Subject-Dependent 4-Class Temporal SNN",
        "evaluation_type": "subject_dependent_stratified_kfold",
        "fast_mode": fast_mode,
        "max_subjects": max_subjects if fast_mode else None,
        "subjects_used": subject_ids,
        "label_strategy": label_strategy,
        "model_config": BEST_TEMPORAL_SNN_CONFIG,
        "features": {
            "type": "windowed_differential_entropy",
            "num_windows": TEMPORAL_NUM_WINDOWS,
            "features_per_window": 200,
        },
        "cross_validation": {
            "n_splits": N_SPLITS,
            "shuffle": True,
            "random_state": CV_RANDOM_STATE,
            "min_class_samples_required": MIN_CLASS_SAMPLES,
        },
        "reference_pooled_baseline": DEAP_TEMPORAL_BASELINE,
        "summary": summary,
        "subject_results": subject_results,
        "skipped_subjects": skipped_subjects,
    }

    if fast_mode and summary:
        payload["fast_run_recommendation"] = {
            "threshold_accuracy": FULL_RUN_RECOMMEND_ACC,
            "threshold_macro_f1": FULL_RUN_RECOMMEND_MACRO_F1,
            "recommend_full_run": (
                summary["overall_mean_accuracy"] > FULL_RUN_RECOMMEND_ACC
                or summary["overall_mean_macro_f1"] > FULL_RUN_RECOMMEND_MACRO_F1
            ),
        }

    _save_results(payload)
    return payload
