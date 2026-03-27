from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Literal, Optional, TypedDict, Union


@dataclass
class LiftTestData:
    control_conversions: int
    treatment_conversions: int
    control_size: int
    treatment_size: int


@dataclass
class GeoData:
    treatment_pre_conversions: float
    treatment_post_conversions: float
    control_pre_conversions: float
    control_post_conversions: float
    treatment_regions: int
    control_regions: int


@dataclass
class CampaignObserved:
    campaign_id: str
    impressions: int
    clicks: int
    conversions: int
    cost: float
    avg_conversion_value: float
    has_lift_test: bool
    lift_test_data: Optional[LiftTestData]
    has_geo_experiment: bool
    geo_data: Optional[GeoData]

    def to_agent_dict(self) -> Dict[str, Any]:
        return {
            "campaign_id": self.campaign_id,
            "impressions": self.impressions,
            "clicks": self.clicks,
            "conversions": self.conversions,
            "cost": round(self.cost, 2),
            "avg_conversion_value": round(self.avg_conversion_value, 2),
            "has_lift_test": self.has_lift_test,
            "lift_test_data": asdict(self.lift_test_data) if self.lift_test_data else None,
            "has_geo_experiment": self.has_geo_experiment,
            "geo_data": asdict(self.geo_data) if self.geo_data else None,
        }


@dataclass
class CampaignHidden:
    baseline_conversions: float
    true_incremental_conversions: float
    true_iroas: float
    availability_type: Literal["both", "lift_only", "geo_only", "neither"]


@dataclass
class Campaign:
    observed: CampaignObserved
    hidden: CampaignHidden

    def tracing_metadata(self) -> Dict[str, Any]:
        return {
            "campaign_id": self.observed.campaign_id,
            "has_lift_test": self.observed.has_lift_test,
            "has_geo_experiment": self.observed.has_geo_experiment,
            "true_iROAS": round(self.hidden.true_iroas, 4),
            "data_availability_type": self.hidden.availability_type,
        }


@dataclass
class ToolResult:
    tool_name: str
    estimated_incremental_conversions: float
    estimated_iroas: float
    diagnostics: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tool_name": self.tool_name,
            "estimated_incremental_conversions": round(self.estimated_incremental_conversions, 4),
            "estimated_iROAS": round(self.estimated_iroas, 4),
            "diagnostics": self.diagnostics,
        }


@dataclass
class FinalAnswer:
    estimated_incremental_conversions: float
    estimated_iroas: float
    explanation: str
    final_estimator_used: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "estimated_incremental_conversions": round(self.estimated_incremental_conversions, 4),
            "estimated_iROAS": round(self.estimated_iroas, 4),
            "explanation": self.explanation,
            "final_estimator_used": self.final_estimator_used,
        }


class StepRecord(TypedDict):
    thought: str
    action: str
    action_input: Union[Dict[str, Any], str]
    observation: Optional[Dict[str, Any]]
    raw_text: str


class AgentState(TypedDict, total=False):
    campaign: Campaign
    selected_campaign: Dict[str, Any]
    available_tools: List[str]
    prompt_text: str
    trajectory: List[StepRecord]
    evidence_log: List[Dict[str, Any]]
    final_answer: FinalAnswer
    step_count: int
    max_steps: int
    last_action: str
    last_action_input: Union[Dict[str, Any], str]
    last_thought: str
    last_raw_text: str
    last_tool_name: str
    last_observation: Dict[str, Any]
    last_tool_observation: Dict[str, Any]
    last_observation_summary: str
    last_diagnostics_result: Dict[str, Any]
    last_rct_result: Dict[str, Any]
    last_geo_result: Dict[str, Any]
    last_observational_result: Dict[str, Any]
    final_reasoning_summary: str
