from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from sklearn.feature_selection import f_classif
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

from src.archive.deap_research.deap_cnn_snn import DEAP_TEMPORAL_BASELINE
from src.features import (
    ASYMMETRY_DE_BANDS,
    NUM_SYMMETRIC_DIFFERENCE_CHANNELS,
    SYMMETRIC_DIFFERENCE_PAIRS,
    TEMPORAL_NUM_WINDOWS,
    extract_temporal_window_de_features,
    extract_temporal_window_symmetric_de_features,
    print_symmetric_difference_feature_info,
)
from src.labels import EMOTION_LABELS, create_multi_emotion_labels, print_class_distribution
from src.snn_model import BEST_TEMPORAL_SNN_CONFIG, train_tuned_snn_model

METRICS_JSON = Path("results/metrics/deap_asymmetry_snn_results.json")
METRICS_CSV = Path("results/metrics/deap_asymmetry_snn_results.csv")


def _evaluate_multiclass(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    model_name: str,
) -> Dict[str, Any]:
    label_ids = list(range(4))
    target_names = [EMOTION_LABELS[i] for i in label_ids]

    acc = float(accuracy_score(y_true, y_pred))
    macro_f1 = float(
        f1_score(y_true, y_pred, average="macro", labels=label_ids, zero_division=0)
    )
    weighted_f1 = float(
        f1_score(y_true, y_pred, average="weighted", labels=label_ids, zero_division=0)
    )
    cm = confusion_matrix(y_true, y_pred, labels=label_ids)
    report_str = classification_report(
        y_true,
        y_pred,
        labels=label_ids,
        target_names=target_names,
        zero_division=0,
    )
    report_dict = classification_report(
        y_true,
        y_pred,
        labels=label_ids,
        target_names=target_names,
        zero_division=0,
        output_dict=True,
    )

    print(f"\n=== {model_name} ===")
    print("Accuracy:", f"{acc:.4f}")
    print("Macro F1:", f"{macro_f1:.4f}")
    print("Weighted F1:", f"{weighted_f1:.4f}")
    print("Confusion matrix:")
    print(cm)
    print("Classification report:")
    print(report_str)

    return {
        "accuracy": acc,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "confusion_matrix": cm.tolist(),
        "classification_report": report_dict,
    }


def _run_temporal_snn_experiment(
    X_temporal: np.ndarray,
    y_multi: np.ndarray,
    *,
    experiment_name: str,
) -> Dict[str, Any]:
    print(f"\n{'=' * 60}")
    print(experiment_name)
    print("=" * 60)
    print("Temporal feature shape:", X_temporal.shape)
    print("Model config:", BEST_TEMPORAL_SNN_CONFIG)

    (
        _model,
        _X_test,
        y_test,
        y_pred,
        accuracy,
        macro_f1,
        params,
    ) = train_tuned_snn_model(
        X_temporal,
        y_multi,
        temporal=True,
        use_best_temporal_config=True,
        temporal_spike_encoding=False,
        quiet=False,
    )

    metrics = _evaluate_multiclass(
        y_test,
        y_pred,
        model_name=f"Temporal SNN — {experiment_name}",
    )
    metrics["params"] = params
    metrics["feature_shape"] = list(X_temporal.shape)
    return metrics


def _analyze_asymmetry_channels(
    X_asym_temporal: np.ndarray,
    y_multi: np.ndarray,
) -> List[Dict[str, Any]]:
    """
    Per asymmetry channel: mean/std/var of DE features + ANOVA F-score vs labels.
    """
    n_trials, n_windows, flat = X_asym_temporal.shape
    n_bands = len(ASYMMETRY_DE_BANDS)
    n_channels = NUM_SYMMETRIC_DIFFERENCE_CHANNELS

    X_ch = X_asym_temporal.reshape(n_trials, n_windows, n_channels, n_bands)

    channel_stats: List[Dict[str, Any]] = []

    print("\n=== Asymmetry Channel ANOVA Analysis ===")

    for ch_idx in range(n_channels):
        left_name, right_name = SYMMETRIC_DIFFERENCE_PAIRS[ch_idx]
        pair_label = f"{left_name}-{right_name}"

        ch_data = X_ch[:, :, ch_idx, :].reshape(n_trials, -1)
        ch_mean = float(np.mean(ch_data))
        ch_std = float(np.std(ch_data))
        ch_var = float(np.var(ch_data))

        f_scores, _p_values = f_classif(ch_data, y_multi)
        mean_f = float(np.mean(f_scores))
        max_f = float(np.max(f_scores))

        channel_stats.append(
            {
                "rank": 0,
                "channel_pair": pair_label,
                "left_channel": left_name,
                "right_channel": right_name,
                "mean": ch_mean,
                "std": ch_std,
                "variance": ch_var,
                "mean_f_score": mean_f,
                "max_f_score": max_f,
            }
        )

    channel_stats.sort(key=lambda x: x["mean_f_score"], reverse=True)
    for rank, entry in enumerate(channel_stats, start=1):
        entry["rank"] = rank

    print("\nTop 10 most informative asymmetry channels (by mean ANOVA F-score):")
    print(f"{'Rank':<6} {'Channel Pair':<12} {'Mean':>10} {'Std':>10} {'Variance':>12} {'Mean F':>10}")
    for entry in channel_stats[:10]:
        print(
            f"{entry['rank']:<6} {entry['channel_pair']:<12} "
            f"{entry['mean']:>10.4f} {entry['std']:>10.4f} "
            f"{entry['variance']:>12.4f} {entry['mean_f_score']:>10.4f}"
        )

    return channel_stats


def _print_comparison(
    baseline: Dict[str, Any],
    asymmetry: Dict[str, Any],
    *,
    reference: Dict[str, float],
) -> None:
    print("\n=== Step 45 Comparison ===")
    print(f"{'Model':<42} {'Accuracy':>10} {'Macro F1':>10} {'Weighted F1':>12}")
    print(
        f"{'Baseline Temporal SNN (this run)':<42} "
        f"{baseline['accuracy']:>10.4f} {baseline['macro_f1']:>10.4f} {baseline['weighted_f1']:>12.4f}"
    )
    print(
        f"{'Symmetric Difference Temporal SNN':<42} "
        f"{asymmetry['accuracy']:>10.4f} {asymmetry['macro_f1']:>10.4f} {asymmetry['weighted_f1']:>12.4f}"
    )
    print(
        f"{'Historical reference (Step 29)':<42} "
        f"{reference['accuracy']:>10.4f} {reference['macro_f1']:>10.4f} {'—':>12}"
    )

    acc_delta = asymmetry["accuracy"] - baseline["accuracy"]
    f1_delta = asymmetry["macro_f1"] - baseline["macro_f1"]
    print("\nSymmetric Difference vs Baseline (this run):")
    print(f"  Accuracy delta:  {acc_delta:+.4f}")
    print(f"  Macro F1 delta:  {f1_delta:+.4f}")

    ref_acc_delta = asymmetry["accuracy"] - reference["accuracy"]
    ref_f1_delta = asymmetry["macro_f1"] - reference["macro_f1"]
    print("\nSymmetric Difference vs Historical reference (53.12% / 0.5103):")
    print(f"  Accuracy delta:  {ref_acc_delta:+.4f}")
    print(f"  Macro F1 delta:  {ref_f1_delta:+.4f}")

    improved = (
        asymmetry["accuracy"] > reference["accuracy"]
        or asymmetry["macro_f1"] > reference["macro_f1"]
    )
    if improved:
        print(
            "\nConclusion: Asymmetry features improved over the historical baseline. "
            "Asymmetry representation may replace the original 40-channel input."
        )
    else:
        print(
            "\nConclusion: No improvement over historical baseline. "
            "DEAP performance limitation is likely dominated by label ambiguity "
            "rather than feature representation."
        )


def _save_results(payload: Dict[str, Any]) -> None:
    METRICS_JSON.parent.mkdir(parents=True, exist_ok=True)

    with open(METRICS_JSON, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    rows = []
    for exp in payload.get("experiments", []):
        rows.append(
            {
                "experiment": exp["name"],
                "feature_shape": json.dumps(exp.get("feature_shape", [])),
                "accuracy": exp["accuracy"],
                "macro_f1": exp["macro_f1"],
                "weighted_f1": exp["weighted_f1"],
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


def run_deap_asymmetry_snn(
    folder: str = "data/raw",
    *,
    label_strategy: str = "mean",
    trials_per_subject: int = 40,
) -> Dict[str, Any]:
    """
    Step 45: compare baseline 40-channel temporal DE vs 14 symmetric-difference channels.
    Always uses full DEAP (32 subjects, 1280 trials).
    """
    from src.load_data import load_all_deap_files
    from src.preprocessing import bandpass_filter

    print("\n" + "=" * 60)
    print("DEAP Symmetric Difference Temporal SNN (Step 45)")
    print("=" * 60)
    print("FAST_TEST_MODE overridden — using full DEAP dataset")
    print("Label strategy:", label_strategy)
    print("Reference baseline:", DEAP_TEMPORAL_BASELINE)

    X, y_ratings = load_all_deap_files(folder, max_subjects=None)
    X_filtered = bandpass_filter(X)

    n_subjects = X_filtered.shape[0] // trials_per_subject
    n_trials = X_filtered.shape[0]

    y_multi = create_multi_emotion_labels(y_ratings, strategy=label_strategy, verbose=False)

    print("\n=== Dataset Summary ===")
    print("Number of subjects:", n_subjects)
    print("Number of trials:", n_trials)
    print("Class distribution:")
    print_class_distribution(y_multi, EMOTION_LABELS, num_classes=4)

    X_baseline = extract_temporal_window_de_features(
        X_filtered,
        num_windows=TEMPORAL_NUM_WINDOWS,
    )
    X_asymmetry = extract_temporal_window_symmetric_de_features(
        X_filtered,
        num_windows=TEMPORAL_NUM_WINDOWS,
    )
    print_symmetric_difference_feature_info(
        X_baseline,
        X_asymmetry,
        num_windows=TEMPORAL_NUM_WINDOWS,
    )
    print("Temporal feature shape (baseline):", X_baseline.shape)
    print("Temporal feature shape (asymmetry):", X_asymmetry.shape)

    baseline_metrics = _run_temporal_snn_experiment(
        X_baseline,
        y_multi,
        experiment_name="Experiment 1 — Baseline Temporal SNN (10 × 200)",
    )
    asymmetry_metrics = _run_temporal_snn_experiment(
        X_asymmetry,
        y_multi,
        experiment_name="Experiment 2 — Symmetric Difference Temporal SNN (10 × 70)",
    )

    channel_analysis = _analyze_asymmetry_channels(X_asymmetry, y_multi)

    _print_comparison(
        baseline_metrics,
        asymmetry_metrics,
        reference=DEAP_TEMPORAL_BASELINE,
    )

    improved_vs_reference = (
        asymmetry_metrics["accuracy"] > DEAP_TEMPORAL_BASELINE["accuracy"]
        or asymmetry_metrics["macro_f1"] > DEAP_TEMPORAL_BASELINE["macro_f1"]
    )

    payload: Dict[str, Any] = {
        "study": "Step 45 DEAP Symmetric Difference Temporal SNN",
        "dataset": {
            "subjects": n_subjects,
            "trials": n_trials,
            "trials_per_subject": trials_per_subject,
        },
        "label_strategy": label_strategy,
        "reference_baseline": DEAP_TEMPORAL_BASELINE,
        "model_config": BEST_TEMPORAL_SNN_CONFIG,
        "symmetric_difference_pairs": [
            {"left": left, "right": right, "formula": f"{left}-{right}"}
            for left, right in SYMMETRIC_DIFFERENCE_PAIRS
        ],
        "asymmetry_bands_hz": [list(b) for b in ASYMMETRY_DE_BANDS],
        "experiments": [
            {
                "name": "Baseline Temporal SNN",
                "description": "10 windows × 200 features (40 ch × 5 DE bands)",
                **baseline_metrics,
            },
            {
                "name": "Symmetric Difference Temporal SNN",
                "description": "10 windows × 70 features (14 asym ch × 5 bands)",
                **asymmetry_metrics,
            },
        ],
        "comparison": {
            "accuracy_delta_asym_vs_baseline_run": asymmetry_metrics["accuracy"]
            - baseline_metrics["accuracy"],
            "macro_f1_delta_asym_vs_baseline_run": asymmetry_metrics["macro_f1"]
            - baseline_metrics["macro_f1"],
            "accuracy_delta_asym_vs_reference": asymmetry_metrics["accuracy"]
            - DEAP_TEMPORAL_BASELINE["accuracy"],
            "macro_f1_delta_asym_vs_reference": asymmetry_metrics["macro_f1"]
            - DEAP_TEMPORAL_BASELINE["macro_f1"],
            "improved_vs_reference": improved_vs_reference,
        },
        "asymmetry_channel_anova": channel_analysis,
        "conclusion": (
            "Asymmetry features should replace the original 40-channel representation."
            if improved_vs_reference
            else "DEAP performance limitation is likely dominated by label ambiguity rather than feature representation."
        ),
    }

    _save_results(payload)
    return payload
