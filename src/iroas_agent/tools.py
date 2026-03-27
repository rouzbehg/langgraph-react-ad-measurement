from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

from .schemas import Campaign, ToolResult
from .tracing import traced


def _safe_iroas(incremental_conversions: float, avg_conversion_value: float, cost: float) -> float:
    return (incremental_conversions * avg_conversion_value) / max(cost, 1.0)


@traced(name="diagnostics_tool", run_type="tool")
def diagnostics_tool(campaign: Campaign, _: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    observed = campaign.observed
    lift_sample_size = 0
    if observed.lift_test_data:
        lift_sample_size = observed.lift_test_data.control_size + observed.lift_test_data.treatment_size

    geo_regions = 0
    if observed.geo_data:
        geo_regions = observed.geo_data.control_regions + observed.geo_data.treatment_regions

    confidence = "low"
    if observed.has_lift_test and lift_sample_size >= 50_000:
        confidence = "high"
    elif observed.has_geo_experiment and geo_regions >= 10:
        confidence = "medium"
    elif observed.conversions >= 100:
        confidence = "medium"

    return {
        "available_data_types": {
            "lift_test": observed.has_lift_test,
            "geo_experiment": observed.has_geo_experiment,
            "aggregate_observational": True,
        },
        "sample_sizes": {
            "impressions": observed.impressions,
            "clicks": observed.clicks,
            "conversions": observed.conversions,
            "lift_test_sample_size": lift_sample_size,
            "geo_region_count": geo_regions,
        },
        "confidence_indicators": {
            "overall_confidence": confidence,
            "rct_reliability": "high" if observed.has_lift_test and lift_sample_size >= 50_000 else "medium" if observed.has_lift_test else "unavailable",
            "geo_reliability": "medium" if observed.has_geo_experiment else "unavailable",
            "observational_reliability": "low",
        },
    }


@traced(name="rct_estimator_tool", run_type="tool")
def rct_estimator_tool(campaign: Campaign, _: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    observed = campaign.observed
    if not observed.lift_test_data:
        raise ValueError("Lift test data is unavailable for this campaign.")

    lift = observed.lift_test_data
    control_rate = lift.control_conversions / max(lift.control_size, 1)
    treatment_rate = lift.treatment_conversions / max(lift.treatment_size, 1)
    estimated_incremental_conversions = (treatment_rate - control_rate) * observed.impressions
    pooled_rate = (lift.control_conversions + lift.treatment_conversions) / max(lift.control_size + lift.treatment_size, 1)
    variance_proxy = pooled_rate * (1 - pooled_rate) * ((1 / max(lift.control_size, 1)) + (1 / max(lift.treatment_size, 1)))
    std_error = math.sqrt(max(variance_proxy, 0.0)) * observed.impressions

    result = ToolResult(
        tool_name="rct_estimator_tool",
        estimated_incremental_conversions=estimated_incremental_conversions,
        estimated_iroas=_safe_iroas(estimated_incremental_conversions, observed.avg_conversion_value, observed.cost),
        diagnostics={
            "variance_proxy": round(variance_proxy, 8),
            "std_error_proxy": round(std_error, 4),
            "sample_size": lift.control_size + lift.treatment_size,
            "control_rate": round(control_rate, 6),
            "treatment_rate": round(treatment_rate, 6),
            "estimator_profile": "low_bias_high_variance",
        },
    )
    return result.to_dict()


@traced(name="geo_diff_in_diff_tool", run_type="tool")
def geo_diff_in_diff_tool(campaign: Campaign, _: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    observed = campaign.observed
    if not observed.geo_data:
        raise ValueError("Geo experiment data is unavailable for this campaign.")

    geo = observed.geo_data
    raw_diff_in_diff = (
        (geo.treatment_post_conversions - geo.treatment_pre_conversions)
        - (geo.control_post_conversions - geo.control_pre_conversions)
    )
    biased_incremental = raw_diff_in_diff * 1.08 + observed.conversions * 0.015
    region_count = geo.treatment_regions + geo.control_regions
    variance_proxy = abs(raw_diff_in_diff) / max(region_count, 1) + observed.conversions * 0.005

    result = ToolResult(
        tool_name="geo_diff_in_diff_tool",
        estimated_incremental_conversions=biased_incremental,
        estimated_iroas=_safe_iroas(biased_incremental, observed.avg_conversion_value, observed.cost),
        diagnostics={
            "variance_proxy": round(variance_proxy, 4),
            "region_count": region_count,
            "raw_diff_in_diff": round(raw_diff_in_diff, 4),
            "bias_note": "moderate_positive_bias_injected",
            "estimator_profile": "moderate_bias_moderate_variance",
        },
    )
    return result.to_dict()


@traced(name="observational_estimator_tool", run_type="tool")
def observational_estimator_tool(campaign: Campaign, _: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    observed = campaign.observed
    conversion_rate = observed.conversions / max(observed.clicks, 1)
    estimated_incremental_conversions = observed.conversions * 0.58 + observed.clicks * conversion_rate * 0.04
    variance_proxy = max(1.0, observed.conversions * 0.015)

    result = ToolResult(
        tool_name="observational_estimator_tool",
        estimated_incremental_conversions=estimated_incremental_conversions,
        estimated_iroas=_safe_iroas(estimated_incremental_conversions, observed.avg_conversion_value, observed.cost),
        diagnostics={
            "variance_proxy": round(variance_proxy, 4),
            "sample_size": observed.clicks,
            "bias_note": "high_bias_low_variance",
            "assumed_attribution_fraction": 0.58,
            "estimator_profile": "high_bias_low_variance",
        },
    )
    return result.to_dict()


TOOL_REGISTRY = {
    "diagnostics_tool": diagnostics_tool,
    "rct_estimator_tool": rct_estimator_tool,
    "geo_diff_in_diff_tool": geo_diff_in_diff_tool,
    "observational_estimator_tool": observational_estimator_tool,
}


def tool_descriptions() -> List[Dict[str, Any]]:
    return [
        {
            "name": "diagnostics_tool",
            "description": "Returns available data types, sample sizes, and confidence indicators.",
            "input_schema": {"type": "object", "properties": {}, "additionalProperties": False},
        },
        {
            "name": "rct_estimator_tool",
            "description": "Uses lift test treatment-control differences to estimate incremental conversions and iROAS.",
            "input_schema": {"type": "object", "properties": {}, "additionalProperties": False},
        },
        {
            "name": "geo_diff_in_diff_tool",
            "description": "Uses geo pre-post treatment-control aggregates for diff-in-diff estimation with moderate bias and variance.",
            "input_schema": {"type": "object", "properties": {}, "additionalProperties": False},
        },
        {
            "name": "observational_estimator_tool",
            "description": "Uses aggregate campaign metrics to produce a biased but low-variance fallback estimate.",
            "input_schema": {"type": "object", "properties": {}, "additionalProperties": False},
        },
    ]
