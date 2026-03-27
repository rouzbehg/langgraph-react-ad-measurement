from __future__ import annotations

from typing import Any, Dict, List, Optional, TypedDict

from langgraph.graph import END, START, StateGraph

from .agent import default_agent
from .data import DEFAULT_DATASET_PATH, load_campaign_dataset


class StudioState(TypedDict, total=False):
    campaign_id: str
    dataset_path: str
    selected_campaign: Dict[str, Any]
    prediction: Dict[str, Any]
    trajectory: List[Dict[str, Any]]
    run_metadata: Dict[str, Any]
    error: str


def _load_campaign(state: StudioState) -> StudioState:
    dataset_path = state.get("dataset_path") or str(DEFAULT_DATASET_PATH)
    campaigns = load_campaign_dataset(DEFAULT_DATASET_PATH.__class__(dataset_path))
    requested_campaign_id = state.get("campaign_id")

    campaign = None
    if requested_campaign_id:
        for candidate in campaigns:
            if candidate.observed.campaign_id == requested_campaign_id:
                campaign = candidate
                break
        if campaign is None:
            available_ids = ", ".join(c.observed.campaign_id for c in campaigns[:10])
            raise ValueError(
                f"Campaign '{requested_campaign_id}' was not found in {dataset_path}. "
                f"Example available ids: {available_ids}"
            )
    else:
        campaign = campaigns[0]

    return {
        **state,
        "campaign_id": campaign.observed.campaign_id,
        "selected_campaign": campaign.observed.to_agent_dict(),
    }


def _run_agent(state: StudioState) -> StudioState:
    dataset_path = state.get("dataset_path") or str(DEFAULT_DATASET_PATH)
    campaigns = load_campaign_dataset(DEFAULT_DATASET_PATH.__class__(dataset_path))
    target_id = state["campaign_id"]

    campaign = next((item for item in campaigns if item.observed.campaign_id == target_id), None)
    if campaign is None:
        raise ValueError(f"Campaign '{target_id}' was not found in {dataset_path}.")

    agent = default_agent()
    result = agent.run_campaign(campaign)
    return {
        **state,
        "prediction": result["prediction"],
        "trajectory": result["trajectory"],
        "run_metadata": result["metadata"],
    }


_builder = StateGraph(StudioState)
_builder.add_node("load_campaign", _load_campaign)
_builder.add_node("run_agent", _run_agent)
_builder.add_edge(START, "load_campaign")
_builder.add_edge("load_campaign", "run_agent")
_builder.add_edge("run_agent", END)
studio_graph = _builder.compile()

