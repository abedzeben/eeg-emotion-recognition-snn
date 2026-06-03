from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


def export_results_summary(
    results: list[dict[str, Any]],
    output_dir: str | Path = "results/metrics",
) -> tuple[Path, Path]:
    """
    Export evaluation summary to CSV and JSON.

    Each result entry should include:
      task, model, accuracy, macro_f1, best_params, notes
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / "results_summary.csv"
    json_path = output_dir / "results_summary.json"

    rows = []
    for entry in results:
        row = {
            "task": entry["task"],
            "model": entry["model"],
            "accuracy": float(entry["accuracy"]),
            "macro_f1": float(entry["macro_f1"]),
            "best_params": entry.get("best_params", {}),
            "notes": entry.get("notes", ""),
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    df["best_params"] = df["best_params"].apply(
        lambda p: json.dumps(p) if isinstance(p, dict) else str(p)
    )
    df.to_csv(csv_path, index=False)

    json_rows = []
    for entry in results:
        json_rows.append(
            {
                "task": entry["task"],
                "model": entry["model"],
                "accuracy": float(entry["accuracy"]),
                "macro_f1": float(entry["macro_f1"]),
                "best_params": entry.get("best_params", {}),
                "notes": entry.get("notes", ""),
            }
        )

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_rows, f, indent=2)

    return csv_path, json_path
