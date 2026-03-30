from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List

import pandas as pd

EXPERIMENT_OUTPUTS_DIR = Path("data/experiment_outputs")


def save_experiment_output(output: Dict[str, Any], output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2, sort_keys=True), encoding="utf-8")
    return output_path


def load_experiment_output(input_path: Path) -> Dict[str, Any]:
    return json.loads(input_path.read_text(encoding="utf-8"))


def load_or_create_experiment_output(
    output_path: Path,
    build_output: Callable[[], Dict[str, Any]],
    refresh: bool = False,
) -> Dict[str, Any]:
    if output_path.exists() and not refresh:
        return load_experiment_output(output_path)
    output = build_output()
    save_experiment_output(output, output_path)
    return output


def list_experiment_outputs(base_dir: Path = EXPERIMENT_OUTPUTS_DIR) -> List[Path]:
    if not base_dir.exists():
        return []
    return sorted(base_dir.glob("*.json"))


def results_frame(results: Iterable[Dict[str, Any]]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for result in results:
        metadata = result.get("metadata", {})
        trajectory = result.get("trajectory", [])
        tool_actions = [step["action"] for step in trajectory if step["action"] != "finish"]
        estimator_actions = [
            action
            for action in tool_actions
            if action in {"rct_estimator_tool", "geo_diff_in_diff_tool", "observational_estimator_tool"}
        ]
        reported_final_estimator = result.get(
            "reported_final_estimator_used",
            result["prediction"]["final_estimator_used"],
        )
        matched_final_estimator = _match_estimator_from_prediction(trajectory, result["prediction"])
        actual_final_estimator = (
            matched_final_estimator
            or (estimator_actions[-1] if estimator_actions else result["prediction"]["final_estimator_used"])
        )
        step_count = metadata.get("number_of_steps", len(trajectory))
        rows.append(
            {
                "campaign_id": result["campaign_id"],
                "availability_type": metadata["data_availability_type"],
                "has_lift_test": metadata["has_lift_test"],
                "has_geo_experiment": metadata["has_geo_experiment"],
                "predicted_iROAS": result["prediction"]["estimated_iROAS"],
                "true_iROAS": metadata["true_iROAS"],
                "error": result["prediction"]["estimated_iROAS"] - metadata["true_iROAS"],
                "abs_error": abs(result["prediction"]["estimated_iROAS"] - metadata["true_iROAS"]),
                "final_estimator_used": actual_final_estimator,
                "reported_final_estimator_used": reported_final_estimator,
                "final_estimator_source": (
                    result.get("final_estimator_source", "matched_prediction")
                    if matched_final_estimator
                    else result.get("final_estimator_source", "trajectory")
                    if estimator_actions
                    else result.get("final_estimator_source", "model_explanation")
                ),
                "step_count": step_count,
                "tool_sequence": " -> ".join(tool_actions),
                "estimator_sequence": " -> ".join(estimator_actions),
                "last_estimator_in_trajectory": estimator_actions[-1] if estimator_actions else None,
                "first_action": trajectory[0]["action"] if trajectory else None,
                "used_multiple_estimators": len(set(estimator_actions)) > 1,
                "estimator_label_matches_trajectory": (
                    reported_final_estimator == estimator_actions[-1]
                    if estimator_actions
                    else reported_final_estimator == actual_final_estimator
                ),
                "trajectory_length": len(trajectory),
                "explanation": result["prediction"]["explanation"],
            }
        )
    return pd.DataFrame(rows)


def selection_matrix(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()
    return (
        frame.groupby(["availability_type", "final_estimator_used"])
        .size()
        .unstack(fill_value=0)
        .sort_index()
    )


def metric_cards(frame: pd.DataFrame) -> Dict[str, float]:
    if frame.empty:
        return {
            "runs": 0.0,
            "mse": 0.0,
            "bias": 0.0,
            "mae": 0.0,
            "avg_steps": 0.0,
            "causal_preference_rate": 0.0,
        }

    preferred = frame.apply(_used_preferred_estimator, axis=1).mean()
    return {
        "runs": float(len(frame)),
        "mse": float((frame["error"] ** 2).mean()),
        "bias": float(frame["error"].mean()),
        "mae": float(frame["abs_error"].mean()),
        "avg_steps": float(frame["step_count"].mean()),
        "causal_preference_rate": float(preferred),
    }


def slice_summary(frame: pd.DataFrame, slice_col: str) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()
    grouped = frame.groupby(slice_col)
    summary = grouped.agg(
        runs=("campaign_id", "count"),
        mse=("error", lambda s: float((s ** 2).mean())),
        bias=("error", "mean"),
        mae=("abs_error", "mean"),
        avg_steps=("step_count", "mean"),
    )
    return summary.sort_index()


def tool_usage(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=["final_estimator_used", "count"])
    return (
        frame["final_estimator_used"]
        .value_counts()
        .rename_axis("final_estimator_used")
        .reset_index(name="count")
    )


def worst_runs(frame: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()
    columns = [
        "campaign_id",
        "availability_type",
        "final_estimator_used",
        "predicted_iROAS",
        "true_iROAS",
        "error",
        "abs_error",
        "step_count",
        "tool_sequence",
    ]
    return frame.sort_values("abs_error", ascending=False)[columns].head(top_n)


def format_trajectory(result: Dict[str, Any] | List[Dict[str, Any]]) -> str:
    trajectory = result if isinstance(result, list) else result.get("trajectory", [])
    lines: List[str] = []
    for idx, step in enumerate(trajectory, start=1):
        lines.append(f"Step {idx}")
        lines.append(f"Thought: {step['thought']}")
        lines.append(f"Action: {step['action']}")
        lines.append(f"Action Input: {step['action_input']}")
        lines.append(f"Observation: {step['observation']}")
        lines.append("-" * 80)
    return "\n".join(lines)


def _used_preferred_estimator(row: pd.Series) -> bool:
    if row["has_lift_test"]:
        return row["final_estimator_used"] == "rct_estimator_tool"
    if row["has_geo_experiment"]:
        return row["final_estimator_used"] == "geo_diff_in_diff_tool"
    return row["final_estimator_used"] == "observational_estimator_tool"


def _match_estimator_from_prediction(trajectory: List[Dict[str, Any]], prediction: Dict[str, Any]) -> str | None:
    pred_inc = prediction.get("estimated_incremental_conversions")
    pred_iroas = prediction.get("estimated_iROAS")
    if pred_inc is None or pred_iroas is None:
        return None

    for step in reversed(trajectory):
        action = step.get("action")
        observation = step.get("observation") or {}
        if action not in {"rct_estimator_tool", "geo_diff_in_diff_tool", "observational_estimator_tool"}:
            continue
        obs_inc = observation.get("estimated_incremental_conversions")
        obs_iroas = observation.get("estimated_iROAS")
        if obs_inc is None or obs_iroas is None:
            continue
        if abs(float(obs_inc) - float(pred_inc)) <= 0.5 and abs(float(obs_iroas) - float(pred_iroas)) <= 0.01:
            return str(action)
    return None
