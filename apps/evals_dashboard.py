from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from iroas_agent.dashboard import (
    EXPERIMENT_OUTPUTS_DIR,
    format_trajectory,
    list_experiment_outputs,
    load_or_create_experiment_output,
    results_frame,
    selection_matrix,
    slice_summary,
    tool_usage,
    worst_runs,
)
from iroas_agent.runner import run_experiment


st.set_page_config(
    page_title="iROAS Evals Dashboard",
    page_icon="chart_with_upwards_trend",
    layout="wide",
)


def _metric_cards(frame: pd.DataFrame) -> dict[str, float]:
    if frame.empty:
        return {"runs": 0.0, "mse": 0.0, "bias": 0.0, "mae": 0.0, "avg_steps": 0.0, "causal_rate": 0.0}
    causal_rate = frame.apply(_used_preferred_estimator, axis=1).mean()
    return {
        "runs": float(len(frame)),
        "mse": float((frame["error"] ** 2).mean()),
        "bias": float(frame["error"].mean()),
        "mae": float(frame["abs_error"].mean()),
        "avg_steps": float(frame["step_count"].mean()),
        "causal_rate": float(causal_rate),
    }


def _used_preferred_estimator(row: pd.Series) -> bool:
    if row["has_lift_test"]:
        return row["final_estimator_used"] == "rct_estimator_tool"
    if row["has_geo_experiment"]:
        return row["final_estimator_used"] == "geo_diff_in_diff_tool"
    return row["final_estimator_used"] == "observational_estimator_tool"


def _render_header() -> None:
    st.title("Local Evals Dashboard")
    st.caption(
        "Inspect saved experiment artifacts, compare estimator-selection behavior, "
        "and drill into individual Thought -> Action -> Observation trajectories."
    )


def _sidebar_controls() -> tuple[Path, str, int, bool]:
    st.sidebar.header("Controls")
    known_artifacts = list_experiment_outputs()
    default_artifact = known_artifacts[0] if known_artifacts else (EXPERIMENT_OUTPUTS_DIR / "latest.json")

    artifact_path = st.sidebar.text_input("Experiment artifact", value=str(default_artifact))
    dataset_path = st.sidebar.text_input("Dataset path", value="data/synthetic_campaigns.csv")
    sample_size = st.sidebar.slider("Sample trajectories", min_value=5, max_value=100, value=20, step=5)
    refresh_experiment = st.sidebar.checkbox("Refresh artifact by rerunning agent", value=False)

    with st.sidebar.expander("Artifact Notes", expanded=False):
        st.markdown(
            "- The app prefers loading a saved experiment artifact.\n"
            "- If the artifact is missing, or refresh is enabled, it will rerun the agent and save a fresh JSON file.\n"
            "- `final_estimator_used` means the estimator whose numeric output matches the final predicted answer."
        )

    return Path(artifact_path), dataset_path, sample_size, refresh_experiment


def _load_output(artifact_path: Path, dataset_path: str, sample_size: int, refresh_experiment: bool) -> dict:
    with st.spinner("Loading experiment artifact..."):
        return load_or_create_experiment_output(
            artifact_path,
            lambda: run_experiment(dataset_path=dataset_path, sample_size=sample_size),
            refresh=refresh_experiment,
        )


def _render_overview(frame: pd.DataFrame) -> None:
    st.subheader("Overview")
    cards = _metric_cards(frame)
    cols = st.columns(6)
    cols[0].metric("Runs", int(cards["runs"]))
    cols[1].metric("MSE", f"{cards['mse']:.4f}")
    cols[2].metric("Bias", f"{cards['bias']:.4f}")
    cols[3].metric("MAE", f"{cards['mae']:.4f}")
    cols[4].metric("Avg Steps", f"{cards['avg_steps']:.2f}")
    cols[5].metric("Causal Preference", f"{cards['causal_rate']:.0%}")


def _render_selection_section(frame: pd.DataFrame) -> None:
    left, right = st.columns([1.25, 1])
    with left:
        st.subheader("Estimator Selection Matrix")
        st.dataframe(selection_matrix(frame), use_container_width=True)
    with right:
        st.subheader("Final Estimator Usage")
        usage = tool_usage(frame).set_index("final_estimator_used")
        st.bar_chart(usage)


def _render_error_section(frame: pd.DataFrame) -> None:
    st.subheader("Error Analysis")
    left, right = st.columns(2)
    with left:
        scatter_df = frame[["true_iROAS", "predicted_iROAS"]].copy()
        st.scatter_chart(scatter_df, x="true_iROAS", y="predicted_iROAS")
    with right:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(frame["error"], bins=20)
        ax.set_title("Residual Distribution")
        ax.set_xlabel("Prediction Error")
        ax.set_ylabel("Count")
        st.pyplot(fig, clear_figure=True)


def _render_slice_section(frame: pd.DataFrame) -> None:
    st.subheader("Availability Breakdown")
    st.dataframe(slice_summary(frame, "availability_type"), use_container_width=True)


def _render_behavior_section(frame: pd.DataFrame) -> None:
    st.subheader("Agent Behavior")
    left, right = st.columns(2)
    with left:
        behavior = pd.DataFrame(
            {
                "metric": [
                    "diagnostics_first_rate",
                    "multi_estimator_rate",
                    "label_matches_trajectory_rate",
                ],
                "value": [
                    (frame["first_action"] == "diagnostics_tool").mean(),
                    frame["used_multiple_estimators"].mean(),
                    frame["estimator_label_matches_trajectory"].mean(),
                ],
            }
        ).set_index("metric")
        st.dataframe(behavior.style.format("{:.2%}"), use_container_width=True)
    with right:
        top_sequences = frame["tool_sequence"].value_counts().head(10).rename_axis("tool_sequence").reset_index(name="count")
        st.dataframe(top_sequences, use_container_width=True)


def _render_mismatch_section(frame: pd.DataFrame) -> None:
    st.subheader("Attribution Mismatches")
    mismatches = frame[~frame["estimator_label_matches_trajectory"]].copy()
    st.caption(
        "These are runs where the model-reported estimator label disagrees with the executed estimator sequence. "
        "The dashboard uses matched final numeric output to choose the trusted final estimator."
    )
    if mismatches.empty:
        st.success("No label mismatches in the current artifact.")
        return
    cols = [
        "campaign_id",
        "availability_type",
        "final_estimator_used",
        "reported_final_estimator_used",
        "final_estimator_source",
        "tool_sequence",
    ]
    st.dataframe(mismatches[cols], use_container_width=True)


def _render_run_detail(results: list[dict], frame: pd.DataFrame) -> None:
    st.subheader("Run Detail Explorer")
    choices = frame["campaign_id"].tolist()
    default_id = worst_runs(frame, top_n=1)["campaign_id"].iloc[0] if not frame.empty else choices[0]
    selected_campaign_id = st.selectbox("Campaign", choices, index=choices.index(default_id))
    selected_run = next(row for row in results if row["campaign_id"] == selected_campaign_id)

    left, right = st.columns([1, 1.2])
    with left:
        st.markdown("**Prediction**")
        st.json(selected_run["prediction"])
        st.markdown("**Metadata**")
        st.json(selected_run["metadata"])
        st.markdown("**Evidence Log**")
        st.json(selected_run.get("evidence_log", []))
    with right:
        st.markdown("**Trajectory**")
        st.code(format_trajectory(selected_run), language="text")


def main() -> None:
    _render_header()
    artifact_path, dataset_path, sample_size, refresh_experiment = _sidebar_controls()
    output = _load_output(artifact_path, dataset_path, sample_size, refresh_experiment)
    results = output["results"]
    frame = results_frame(results)

    st.caption(f"Artifact: `{artifact_path}`")
    _render_overview(frame)
    _render_selection_section(frame)
    _render_slice_section(frame)
    _render_error_section(frame)
    _render_behavior_section(frame)
    _render_mismatch_section(frame)
    _render_run_detail(results, frame)


if __name__ == "__main__":
    main()
