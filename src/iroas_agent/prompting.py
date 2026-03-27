from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Union

from .schemas import FinalAnswer, StepRecord


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROMPT_PATH = PROJECT_ROOT / "prompts" / "react_agent_prompt.md"

ACTION_PATTERN = re.compile(
    r"Thought:\s*(?P<thought>.*?)\n\s*Action:\s*(?P<action>[a-zA-Z0-9_]+)\n\s*Action Input:\s*(?P<input>.*?)(?:\n\s*Final Answer:\s*(?P<final>.*))?\Z",
    re.DOTALL,
)
FINAL_ONLY_PATTERN = re.compile(
    r"\A\s*Final Answer:\s*(?P<final>.*)\Z",
    re.DOTALL,
)
FINAL_ANSWER_PATTERN = re.compile(
    r"\*\s*estimated_incremental_conversions:\s*(?P<incremental>[-+]?\d*\.?\d+)\s*\n"
    r"\*\s*estimated_iROAS:\s*(?P<iroas>[-+]?\d*\.?\d+)\s*\n"
    r"\*\s*explanation:\s*(?P<explanation>.+)\Z",
    re.DOTALL,
)


def load_react_prompt(path: Path = PROMPT_PATH) -> str:
    return path.read_text(encoding="utf-8")


def render_trajectory(trajectory: Iterable[StepRecord]) -> str:
    blocks: List[str] = []
    for step in trajectory:
        blocks.append(f"Thought: {step['thought']}")
        blocks.append("")
        blocks.append(f"Action: {step['action']}")
        blocks.append("")
        action_input = step["action_input"]
        if isinstance(action_input, str):
            rendered_input = action_input
        else:
            rendered_input = json.dumps(action_input, indent=2, sort_keys=True)
        blocks.append(f"Action Input: {rendered_input}")
        if step.get("observation") is not None:
            blocks.append("")
            blocks.append(f"Observation: {json.dumps(step['observation'], indent=2, sort_keys=True)}")
        blocks.append("")
        blocks.append("---")
        blocks.append("")
    return "\n".join(blocks).strip() or "No prior steps."


def build_agent_prompt(prompt_template: str, campaign_dict: Dict[str, Any], tools: List[Dict[str, Any]], trajectory: Iterable[StepRecord]) -> str:
    return prompt_template.format(
        tool_descriptions=json.dumps(tools, indent=2, sort_keys=True),
        campaign_json=json.dumps(campaign_dict, indent=2, sort_keys=True),
        trajectory=render_trajectory(trajectory),
    )


def parse_action_block(text: str) -> Dict[str, Any]:
    stripped = text.strip()
    match = ACTION_PATTERN.search(stripped)
    if not match:
        final_only_match = FINAL_ONLY_PATTERN.search(stripped)
        if final_only_match:
            final_answer = parse_final_answer(final_only_match.group("final").strip(), action="finish")
            return {
                "thought": "The model returned a final answer without the explicit ReAct wrapper.",
                "action": "finish",
                "action_input": "none",
                "raw_text": stripped,
                "final_answer": final_answer,
            }
        raise ValueError(f"LLM output did not match required format.\n{text}")

    raw_input = match.group("input").strip()
    parsed_input: Union[Dict[str, Any], str]
    if raw_input.lower() == "none":
        parsed_input = "none"
    else:
        parsed_input = json.loads(raw_input)

    result = {
        "thought": match.group("thought").strip(),
        "action": match.group("action").strip(),
        "action_input": parsed_input,
        "raw_text": text.strip(),
    }

    final_block = match.group("final")
    if final_block:
        result["final_answer"] = parse_final_answer(final_block.strip(), action=result["action"])
    return result


def parse_final_answer(text: str, action: str = "finish") -> FinalAnswer:
    if action != "finish":
        raise ValueError("Final answer can only be parsed when action is finish.")
    match = FINAL_ANSWER_PATTERN.search(text.strip())
    if not match:
        raise ValueError(f"Final answer block did not match required format.\n{text}")
    explanation = " ".join(match.group("explanation").split())
    return FinalAnswer(
        estimated_incremental_conversions=float(match.group("incremental")),
        estimated_iroas=float(match.group("iroas")),
        explanation=explanation,
        final_estimator_used=_infer_final_estimator(explanation),
    )


def _infer_final_estimator(explanation: str) -> str:
    lowered = explanation.lower()
    if "rct" in lowered or "lift test" in lowered:
        return "rct_estimator_tool"
    if "geo" in lowered or "diff-in-diff" in lowered:
        return "geo_diff_in_diff_tool"
    if "observational" in lowered:
        return "observational_estimator_tool"
    return "unknown"
