from __future__ import annotations

import json
import math
import random
from csv import DictReader
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from .schemas import Campaign, CampaignHidden, CampaignObserved, GeoData, LiftTestData


DEFAULT_DATASET_PATH = Path("data/synthetic_campaigns.csv")


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(value, high))


def _draw_positive(rng: random.Random, mean: float, std_dev: float, floor: float = 0.0) -> float:
    return max(floor, rng.gauss(mean, std_dev))


def _choose_availability(rng: random.Random) -> str:
    draw = rng.random()
    if draw < 0.2:
        return "both"
    if draw < 0.45:
        return "lift_only"
    if draw < 0.7:
        return "geo_only"
    return "neither"


def generate_campaigns(count: int, seed: int = 7) -> List[Campaign]:
    rng = random.Random(seed)
    return [generate_campaign(index=i, rng=rng) for i in range(count)]


def generate_campaign(index: int, rng: Optional[random.Random] = None) -> Campaign:
    rng = rng or random.Random(index)
    campaign_id = f"campaign_{index:04d}"

    impressions = int(rng.randint(80_000, 2_000_000))
    ctr = rng.uniform(0.007, 0.035)
    clicks = max(100, int(impressions * ctr + rng.gauss(0, impressions * 0.0015)))

    baseline_conversion_rate = rng.uniform(0.015, 0.09)
    incremental_rate_per_click = rng.uniform(0.0, 0.06)

    baseline_conversions = clicks * baseline_conversion_rate
    true_incremental_conversions = max(0.0, clicks * incremental_rate_per_click + rng.gauss(0, clicks * 0.002))
    noise = rng.gauss(0, math.sqrt(max(baseline_conversions + true_incremental_conversions, 1.0)) * 0.75)
    observed_conversions = max(0, int(round(baseline_conversions + true_incremental_conversions + noise)))

    cpc = rng.uniform(0.8, 4.5)
    cost = max(500.0, clicks * cpc + rng.gauss(0, clicks * 0.1))
    avg_conversion_value = rng.uniform(40.0, 250.0)
    true_iroas = (true_incremental_conversions * avg_conversion_value) / max(cost, 1.0)

    availability_type = _choose_availability(rng)
    has_lift_test = availability_type in {"both", "lift_only"}
    has_geo_experiment = availability_type in {"both", "geo_only"}

    lift_test_data = _generate_lift_test_data(
        rng=rng,
        impressions=impressions,
        baseline_rate_per_impression=baseline_conversions / impressions,
        incremental_rate_per_impression=true_incremental_conversions / impressions,
        enabled=has_lift_test,
    )
    geo_data = _generate_geo_data(
        rng=rng,
        total_incremental_conversions=true_incremental_conversions,
        baseline_conversions=baseline_conversions,
        enabled=has_geo_experiment,
    )

    observed = CampaignObserved(
        campaign_id=campaign_id,
        impressions=impressions,
        clicks=clicks,
        conversions=observed_conversions,
        cost=cost,
        avg_conversion_value=avg_conversion_value,
        has_lift_test=has_lift_test,
        lift_test_data=lift_test_data,
        has_geo_experiment=has_geo_experiment,
        geo_data=geo_data,
    )
    hidden = CampaignHidden(
        baseline_conversions=baseline_conversions,
        true_incremental_conversions=true_incremental_conversions,
        true_iroas=true_iroas,
        availability_type=availability_type,  # type: ignore[arg-type]
    )
    return Campaign(observed=observed, hidden=hidden)


def _generate_lift_test_data(
    rng: random.Random,
    impressions: int,
    baseline_rate_per_impression: float,
    incremental_rate_per_impression: float,
    enabled: bool,
) -> Optional[LiftTestData]:
    if not enabled:
        return None

    sample_fraction = rng.uniform(0.05, 0.2)
    total_sample = max(10_000, int(impressions * sample_fraction))
    control_size = total_sample // 2
    treatment_size = total_sample - control_size

    control_mean = baseline_rate_per_impression * control_size
    treatment_mean = (baseline_rate_per_impression + incremental_rate_per_impression) * treatment_size

    control_noise = rng.gauss(0, math.sqrt(max(control_mean, 1.0)) * 1.2)
    treatment_noise = rng.gauss(0, math.sqrt(max(treatment_mean, 1.0)) * 1.2)

    return LiftTestData(
        control_conversions=max(0, int(round(control_mean + control_noise))),
        treatment_conversions=max(0, int(round(treatment_mean + treatment_noise))),
        control_size=control_size,
        treatment_size=treatment_size,
    )


def _generate_geo_data(
    rng: random.Random,
    total_incremental_conversions: float,
    baseline_conversions: float,
    enabled: bool,
) -> Optional[GeoData]:
    if not enabled:
        return None

    treatment_regions = rng.randint(4, 10)
    control_regions = rng.randint(4, 10)
    pre_scale = rng.uniform(0.42, 0.58)

    control_pre = _draw_positive(rng, baseline_conversions * pre_scale * 0.45, baseline_conversions * 0.03, floor=1.0)
    control_growth = rng.uniform(-0.03, 0.08)
    control_post = _draw_positive(rng, control_pre * (1 + control_growth), control_pre * 0.04, floor=1.0)

    treatment_pre = _draw_positive(rng, baseline_conversions * pre_scale * 0.55, baseline_conversions * 0.03, floor=1.0)
    treatment_trend = control_growth + rng.uniform(-0.01, 0.03)
    treatment_post_base = treatment_pre * (1 + treatment_trend)
    treatment_post = _draw_positive(
        rng,
        treatment_post_base + total_incremental_conversions * rng.uniform(0.8, 1.15),
        max(3.0, treatment_pre * 0.06),
        floor=1.0,
    )

    return GeoData(
        treatment_pre_conversions=treatment_pre,
        treatment_post_conversions=treatment_post,
        control_pre_conversions=control_pre,
        control_post_conversions=control_post,
        treatment_regions=treatment_regions,
        control_regions=control_regions,
    )


def campaign_rows(campaigns: Iterable[Campaign]) -> List[dict]:
    rows = []
    for campaign in campaigns:
        rows.append(flatten_campaign(campaign))
    return rows


def flatten_campaign(campaign: Campaign) -> dict:
    observed = campaign.observed
    row = observed.to_agent_dict()

    lift = observed.lift_test_data
    geo = observed.geo_data
    row["lift_test_control_conversions"] = lift.control_conversions if lift else None
    row["lift_test_treatment_conversions"] = lift.treatment_conversions if lift else None
    row["lift_test_control_size"] = lift.control_size if lift else None
    row["lift_test_treatment_size"] = lift.treatment_size if lift else None

    row["geo_treatment_pre_conversions"] = geo.treatment_pre_conversions if geo else None
    row["geo_treatment_post_conversions"] = geo.treatment_post_conversions if geo else None
    row["geo_control_pre_conversions"] = geo.control_pre_conversions if geo else None
    row["geo_control_post_conversions"] = geo.control_post_conversions if geo else None
    row["geo_treatment_regions"] = geo.treatment_regions if geo else None
    row["geo_control_regions"] = geo.control_regions if geo else None

    row["baseline_conversions"] = round(campaign.hidden.baseline_conversions, 4)
    row["true_incremental_conversions"] = round(campaign.hidden.true_incremental_conversions, 4)
    row["true_iROAS"] = round(campaign.hidden.true_iroas, 4)
    row["availability_type"] = campaign.hidden.availability_type
    return row


def write_campaign_dataset(campaigns: Iterable[Campaign], output_path: Path) -> Path:
    rows = campaign_rows(campaigns)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.suffix.lower() == ".json":
        output_path.write_text(json.dumps(rows, indent=2, sort_keys=True), encoding="utf-8")
        return output_path

    if output_path.suffix.lower() == ".jsonl":
        payload = "\n".join(json.dumps(row, sort_keys=True) for row in rows)
        output_path.write_text(payload + ("\n" if payload else ""), encoding="utf-8")
        return output_path

    if output_path.suffix.lower() == ".csv":
        import csv

        with output_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()) if rows else [])
            if rows:
                writer.writeheader()
                writer.writerows(rows)
        return output_path

    raise ValueError("Unsupported dataset format. Use .csv, .json, or .jsonl.")


def load_campaign_dataset(input_path: Path) -> List[Campaign]:
    suffix = input_path.suffix.lower()
    if suffix == ".csv":
        with input_path.open("r", encoding="utf-8", newline="") as handle:
            rows = list(DictReader(handle))
    elif suffix == ".json":
        rows = json.loads(input_path.read_text(encoding="utf-8"))
    elif suffix == ".jsonl":
        rows = [
            json.loads(line)
            for line in input_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    else:
        raise ValueError("Unsupported dataset format. Use .csv, .json, or .jsonl.")

    return [campaign_from_row(row) for row in rows]


def campaign_from_row(row: Dict[str, Any]) -> Campaign:
    has_lift_test = _to_bool(row.get("has_lift_test"))
    has_geo_experiment = _to_bool(row.get("has_geo_experiment"))

    lift_test_data = None
    if has_lift_test:
        lift_test_data = LiftTestData(
            control_conversions=_to_int(row.get("lift_test_control_conversions")),
            treatment_conversions=_to_int(row.get("lift_test_treatment_conversions")),
            control_size=_to_int(row.get("lift_test_control_size")),
            treatment_size=_to_int(row.get("lift_test_treatment_size")),
        )

    geo_data = None
    if has_geo_experiment:
        geo_data = GeoData(
            treatment_pre_conversions=_to_float(row.get("geo_treatment_pre_conversions")),
            treatment_post_conversions=_to_float(row.get("geo_treatment_post_conversions")),
            control_pre_conversions=_to_float(row.get("geo_control_pre_conversions")),
            control_post_conversions=_to_float(row.get("geo_control_post_conversions")),
            treatment_regions=_to_int(row.get("geo_treatment_regions")),
            control_regions=_to_int(row.get("geo_control_regions")),
        )

    observed = CampaignObserved(
        campaign_id=str(row["campaign_id"]),
        impressions=_to_int(row.get("impressions")),
        clicks=_to_int(row.get("clicks")),
        conversions=_to_int(row.get("conversions")),
        cost=_to_float(row.get("cost")),
        avg_conversion_value=_to_float(row.get("avg_conversion_value")),
        has_lift_test=has_lift_test,
        lift_test_data=lift_test_data,
        has_geo_experiment=has_geo_experiment,
        geo_data=geo_data,
    )
    hidden = CampaignHidden(
        baseline_conversions=_to_float(row.get("baseline_conversions")),
        true_incremental_conversions=_to_float(row.get("true_incremental_conversions")),
        true_iroas=_to_float(row.get("true_iROAS")),
        availability_type=str(row["availability_type"]),  # type: ignore[arg-type]
    )
    return Campaign(observed=observed, hidden=hidden)


def _to_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"true", "1", "yes"}


def _to_int(value: Any) -> int:
    if value in (None, ""):
        return 0
    return int(float(value))


def _to_float(value: Any) -> float:
    if value in (None, ""):
        return 0.0
    return float(value)
