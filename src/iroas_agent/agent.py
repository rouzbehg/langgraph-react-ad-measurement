from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from .prompting import build_agent_prompt, load_react_prompt, parse_action_block
from .schemas import AgentState, Campaign, FinalAnswer, StepRecord
from .tools import TOOL_REGISTRY, tool_descriptions
from .tracing import attach_run_metadata, build_langsmith_extra, traced

try:
    from langgraph.graph import END, START, StateGraph
except ImportError as exc:
    StateGraph = None  # type: ignore[assignment]
    START = "__start__"  # type: ignore[assignment]
    END = "__end__"  # type: ignore[assignment]
    _LANGGRAPH_IMPORT_ERROR = exc
else:
    _LANGGRAPH_IMPORT_ERROR = None

try:
    from langchain_openai import ChatOpenAI
except ImportError:
    ChatOpenAI = None  # type: ignore[assignment]


class ReActModelProtocol:
    def invoke(self, prompt: str, **kwargs: Any) -> str:
        raise NotImplementedError


@dataclass
class OpenAIReActModel(ReActModelProtocol):
    model_name: str = "gpt-4o-mini"
    temperature: float = 0.0

    def __post_init__(self) -> None:
        if ChatOpenAI is None:
            raise ImportError("langchain-openai is required to use OpenAIReActModel.")
        self._client = ChatOpenAI(model=self.model_name, temperature=self.temperature)

    def invoke(self, prompt: str, **kwargs: Any) -> str:
        response = self._client.invoke(prompt, **kwargs)
        return getattr(response, "content", str(response))


class IROASReActAgent:
    def __init__(
        self,
        model: ReActModelProtocol,
        prompt_path: Optional[str] = None,
        max_steps: int = 5,
    ) -> None:
        if StateGraph is None:
            raise ImportError(f"langgraph is required to use IROASReActAgent: {_LANGGRAPH_IMPORT_ERROR}")
        self.model = model
        self.max_steps = max_steps
        self.prompt_text = load_react_prompt() if prompt_path is None else load_react_prompt(path=Path(prompt_path))
        self.graph = self._build_graph()

    def _build_graph(self):
        graph = StateGraph(AgentState)
        graph.add_node("llm_step", self._llm_step)
        graph.add_node("review_observation", self._review_observation)
        for tool_name in TOOL_REGISTRY:
            graph.add_node(tool_name, self._make_tool_node(tool_name))
        graph.add_edge(START, "llm_step")
        graph.add_conditional_edges(
            "llm_step",
            self._route_after_llm,
            {
                "diagnostics_tool": "diagnostics_tool",
                "rct_estimator_tool": "rct_estimator_tool",
                "geo_diff_in_diff_tool": "geo_diff_in_diff_tool",
                "observational_estimator_tool": "observational_estimator_tool",
                "finish": END,
            },
        )
        for tool_name in TOOL_REGISTRY:
            graph.add_edge(tool_name, "review_observation")
        graph.add_edge("review_observation", "llm_step")
        return graph.compile()

    def _route_after_llm(self, state: AgentState) -> str:
        if state["last_action"] == "finish":
            return "finish"
        return state["last_action"]

    @traced(name="react_llm_step", run_type="llm")
    def _llm_step(self, state: AgentState) -> AgentState:
        if state["step_count"] >= state["max_steps"]:
            final_answer = self._fallback_finish(state, "Reached max steps before explicit finish.")
            return {
                **state,
                "last_action": "finish",
                "final_answer": final_answer,
            }

        prompt = build_agent_prompt(
            prompt_template=state["prompt_text"],
            campaign_dict=state["campaign"].observed.to_agent_dict(),
            tools=tool_descriptions(),
            trajectory=state.get("trajectory", []),
        )
        response_text = self.model.invoke(prompt)
        parsed = parse_action_block(response_text)

        next_state: AgentState = {
            **state,
            "step_count": state["step_count"] + 1,
            "last_action": parsed["action"],
            "last_action_input": parsed["action_input"],
            "last_thought": parsed["thought"],
            "last_raw_text": parsed["raw_text"],
        }

        if parsed["action"] == "finish":
            final_answer = parsed.get("final_answer") or self._fallback_finish(state, "Model finished without final answer payload.")
            next_state["final_answer"] = final_answer
            next_state["final_reasoning_summary"] = final_answer.explanation
            step: StepRecord = {
                "thought": parsed["thought"],
                "action": parsed["action"],
                "action_input": parsed["action_input"],
                "observation": None,
                "raw_text": parsed["raw_text"],
            }
            next_state["trajectory"] = [*state.get("trajectory", []), step]
        return next_state

    @traced(name="react_review_observation", run_type="chain")
    def _review_observation(self, state: AgentState) -> AgentState:
        tool_name = state.get("last_tool_name", state.get("last_action", "unknown"))
        observation = state.get("last_tool_observation", state.get("last_observation", {}))
        summary = self._summarize_observation(tool_name, observation)
        evidence_entry = {
            "tool": tool_name,
            "summary": summary,
            "observation": observation,
        }
        return {
            **state,
            "last_observation_summary": summary,
            "evidence_log": [*state.get("evidence_log", []), evidence_entry],
        }

    def _make_tool_node(self, tool_name: str) -> Callable[[AgentState], AgentState]:
        @traced(name=f"react::{tool_name}", run_type="chain")
        def _tool_node(state: AgentState) -> AgentState:
            if tool_name not in TOOL_REGISTRY:
                raise ValueError(f"Unknown tool requested: {tool_name}")

            raw_input = state.get("last_action_input", "none")
            tool_input = {} if raw_input == "none" else raw_input
            observation = TOOL_REGISTRY[tool_name](state["campaign"], tool_input)

            step: StepRecord = {
                "thought": state["last_thought"],
                "action": tool_name,
                "action_input": raw_input,
                "observation": observation,
                "raw_text": state["last_raw_text"],
            }
            return {
                **state,
                "last_tool_name": tool_name,
                "last_observation": observation,
                "last_tool_observation": observation,
                **self._tool_specific_state_update(tool_name, observation),
                "trajectory": [*state.get("trajectory", []), step],
            }

        return _tool_node

    def _tool_specific_state_update(self, tool_name: str, observation: Dict[str, Any]) -> Dict[str, Any]:
        mapping = {
            "diagnostics_tool": "last_diagnostics_result",
            "rct_estimator_tool": "last_rct_result",
            "geo_diff_in_diff_tool": "last_geo_result",
            "observational_estimator_tool": "last_observational_result",
        }
        state_key = mapping.get(tool_name)
        return {state_key: observation} if state_key else {}

    def _summarize_observation(self, tool_name: str, observation: Dict[str, Any]) -> str:
        if tool_name == "diagnostics_tool":
            confidence = observation.get("confidence_indicators", {}).get("overall_confidence", "unknown")
            data_types = observation.get("available_data_types", {})
            available = [name for name, enabled in data_types.items() if enabled]
            return f"Diagnostics found available data sources: {', '.join(available)} with overall confidence {confidence}."

        est_inc = observation.get("estimated_incremental_conversions")
        est_iroas = observation.get("estimated_iROAS")
        diagnostics = observation.get("diagnostics", {})
        profile = diagnostics.get("estimator_profile", "unknown_profile")
        return (
            f"{tool_name} estimated incremental conversions={est_inc}, "
            f"estimated_iROAS={est_iroas}, profile={profile}."
        )

    def _fallback_finish(self, state: AgentState, explanation_prefix: str) -> FinalAnswer:
        trajectory = state.get("trajectory", [])
        estimator_steps = [
            step
            for step in trajectory
            if step["action"] in TOOL_REGISTRY
            and step["action"] != "diagnostics_tool"
            and step.get("observation")
        ]
        if not estimator_steps:
            raise ValueError("Agent finished without any estimator observations.")
        best_step = estimator_steps[-1]
        observation = best_step["observation"] or {}
        return FinalAnswer(
            estimated_incremental_conversions=float(observation["estimated_incremental_conversions"]),
            estimated_iroas=float(observation["estimated_iROAS"]),
            explanation=f"{explanation_prefix} Using the latest estimator output from {best_step['action']}.",
            final_estimator_used=best_step["action"],
        )

    def _latest_estimator_action(self, trajectory: List[StepRecord]) -> Optional[str]:
        for step in reversed(trajectory):
            action = step["action"]
            if action in TOOL_REGISTRY and action != "diagnostics_tool" and step.get("observation"):
                return action
        return None

    def _matched_estimator_action(self, trajectory: List[StepRecord], final_answer: FinalAnswer) -> Optional[str]:
        for step in reversed(trajectory):
            action = step["action"]
            observation = step.get("observation") or {}
            if action not in TOOL_REGISTRY or action == "diagnostics_tool":
                continue
            obs_inc = observation.get("estimated_incremental_conversions")
            obs_iroas = observation.get("estimated_iROAS")
            if obs_inc is None or obs_iroas is None:
                continue
            if (
                abs(float(obs_inc) - final_answer.estimated_incremental_conversions) <= 0.5
                and abs(float(obs_iroas) - final_answer.estimated_iroas) <= 0.01
            ):
                return action
        return None

    @traced(name="iroas_campaign_run", run_type="chain")
    def run_campaign(self, campaign: Campaign) -> Dict[str, Any]:
        metadata = campaign.tracing_metadata()
        initial_state: AgentState = {
            "campaign": campaign,
            "selected_campaign": campaign.observed.to_agent_dict(),
            "available_tools": [tool["name"] for tool in tool_descriptions()],
            "prompt_text": self.prompt_text,
            "trajectory": [],
            "evidence_log": [],
            "step_count": 0,
            "max_steps": self.max_steps,
        }
        result = self.graph.invoke(
            initial_state,
            config={
                "run_name": f"campaign::{campaign.observed.campaign_id}",
                "metadata": metadata,
                "tags": [
                    metadata["data_availability_type"],
                    f"lift={campaign.observed.has_lift_test}",
                    f"geo={campaign.observed.has_geo_experiment}",
                ],
            },
        )
        raw_final_answer = result["final_answer"]
        matched_final_estimator = self._matched_estimator_action(result.get("trajectory", []), raw_final_answer)
        actual_final_estimator = matched_final_estimator or self._latest_estimator_action(result.get("trajectory", []))
        final_answer = raw_final_answer
        if actual_final_estimator and raw_final_answer.final_estimator_used != actual_final_estimator:
            final_answer = FinalAnswer(
                estimated_incremental_conversions=raw_final_answer.estimated_incremental_conversions,
                estimated_iroas=raw_final_answer.estimated_iroas,
                explanation=raw_final_answer.explanation,
                final_estimator_used=actual_final_estimator,
            )
        run_metadata = {
            **metadata,
            "predicted_iROAS": round(final_answer.estimated_iroas, 4),
            "number_of_steps": result["step_count"],
        }
        tags = [
            final_answer.final_estimator_used,
            metadata["data_availability_type"],
            f"steps={result['step_count']}",
        ]
        attach_run_metadata(run_metadata, tags)
        return {
            "campaign_id": campaign.observed.campaign_id,
            "prediction": final_answer.to_dict(),
            "reported_final_estimator_used": raw_final_answer.final_estimator_used,
            "final_estimator_source": (
                "matched_prediction"
                if matched_final_estimator
                else "trajectory"
                if actual_final_estimator
                else "model_explanation"
            ),
            "trajectory": result["trajectory"],
            "metadata": run_metadata,
            "evidence_log": result.get("evidence_log", []),
            "selected_campaign": result.get("selected_campaign"),
            "langsmith_extra": build_langsmith_extra(run_metadata, tags),
        }


def default_agent(model_name: str = "gpt-4o-mini", temperature: float = 0.0, max_steps: int = 5) -> IROASReActAgent:
    return IROASReActAgent(model=OpenAIReActModel(model_name=model_name, temperature=temperature), max_steps=max_steps)
