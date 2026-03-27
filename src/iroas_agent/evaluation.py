from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any, Dict, Iterable, List


def compute_metrics(results: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    rows = list(results)
    if not rows:
        return {"count": 0, "mse": 0.0, "bias": 0.0}

    errors = [row["prediction"]["estimated_iROAS"] - row["metadata"]["true_iROAS"] for row in rows]
    mse = sum(error * error for error in errors) / len(errors)
    bias = sum(errors) / len(errors)

    tool_usage = Counter(row["prediction"]["final_estimator_used"] for row in rows)
    step_counts = [row["metadata"]["number_of_steps"] for row in rows]
    selection_patterns = Counter(
        f"{row['metadata']['data_availability_type']}->{row['prediction']['final_estimator_used']}" for row in rows
    )

    return {
        "count": len(rows),
        "mse": mse,
        "bias": bias,
        "avg_steps": sum(step_counts) / len(step_counts),
        "tool_usage_frequency": dict(tool_usage),
        "selection_patterns": dict(selection_patterns),
    }


def slice_metrics(results: Iterable[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in results:
        grouped[f"lift={row['metadata']['has_lift_test']}"].append(row)
        grouped[f"geo={row['metadata']['has_geo_experiment']}"].append(row)
        grouped[f"availability={row['metadata']['data_availability_type']}"].append(row)
    return {key: compute_metrics(rows) for key, rows in grouped.items()}

