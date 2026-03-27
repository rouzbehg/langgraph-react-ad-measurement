from __future__ import annotations

from typing import Any, Dict, List, TypedDict, Union

from langgraph.graph import END, START, StateGraph

from iroas_agent.agent import default_agent
from iroas_agent.data import DEFAULT_DATASET_PATH, load_campaign_dataset
from iroas_agent.prompting import load_react_prompt
from iroas_agent.schemas import Campaign, FinalAnswer, StepRecord
from iroas_agent.tracing import attach_run_metadata


class StudioState(TypedDict, total=False):
    campaign_id: str
    dataset_path: str
    campaign: Campaign
    prompt_text: str
    trajectory: List[StepRecord]
    final_answer: FinalAnswer
    step_count: int
    max_steps: int
    last_action: str
    last_action_input: Union[Dict[str, Any], str]
    last_thought: str
    last_raw_text: str
    last_observation: Dict[str, Any]
    selected_campaign: Dict[str, Any]
    prediction: Dict[str, Any]
    run_metadata: Dict[str, Any]


_agent = default_agent()


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
        "campaign": campaign,
        "selected_campaign": campaign.observed.to_agent_dict(),
        "prompt_text": load_react_prompt(),
        "trajectory": [],
        "step_count": 0,
        "max_steps": state.get("max_steps", 5),
    }


def _route_after_llm(state: StudioState) -> str:
    if state["last_action"] == "finish":
        return "finalize"
    return "tool_step"


def _finalize(state: StudioState) -> StudioState:
    campaign = state["campaign"]
    final_answer = state["final_answer"]
    metadata = campaign.tracing_metadata()
    run_metadata = {
        **metadata,
        "predicted_iROAS": round(final_answer.estimated_iroas, 4),
        "number_of_steps": state["step_count"],
    }
    tags = [
        final_answer.final_estimator_used,
        metadata["data_availability_type"],
        f"steps={state['step_count']}",
    ]
    attach_run_metadata(run_metadata, tags)
    return {
        **state,
        "prediction": final_answer.to_dict(),
        "run_metadata": run_metadata,
    }


_builder = StateGraph(StudioState)
_builder.add_node("load_campaign", _load_campaign)
_builder.add_node("llm_step", _agent._llm_step)
_builder.add_node("tool_step", _agent._tool_step)
_builder.add_node("finalize", _finalize)
_builder.add_edge(START, "load_campaign")
_builder.add_edge("load_campaign", "llm_step")
_builder.add_conditional_edges(
    "llm_step",
    _route_after_llm,
    {"tool_step": "tool_step", "finalize": "finalize"},
)
_builder.add_edge("tool_step", "llm_step")
_builder.add_edge("finalize", END)
studio_graph = _builder.compile()
