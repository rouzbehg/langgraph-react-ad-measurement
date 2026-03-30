from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from .agent import IROASReActAgent, default_agent
from .dashboard import save_experiment_output
from .data import DEFAULT_DATASET_PATH, generate_campaigns, load_campaign_dataset
from .evaluation import compute_metrics, slice_metrics


def run_experiment(
    campaign_count: int = 25,
    seed: int = 7,
    model_name: str = "gpt-4o-mini",
    max_steps: int = 5,
    sample_size: int = 5,
    dataset_path: Optional[str] = None,
    agent: Optional[IROASReActAgent] = None,
) -> Dict[str, Any]:
    campaigns = load_campaign_dataset(Path(dataset_path)) if dataset_path else generate_campaigns(campaign_count, seed=seed)
    agent = agent or default_agent(model_name=model_name, max_steps=max_steps)

    results: List[Dict[str, Any]] = []
    for campaign in campaigns:
        results.append(agent.run_campaign(campaign))

    summary = compute_metrics(results)
    slices = slice_metrics(results)
    trajectories = results[:sample_size]
    return {
        "summary_metrics": summary,
        "slice_metrics": slices,
        "sample_trajectories": trajectories,
        "results": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the educational LangGraph ReAct iROAS experiment.")
    parser.add_argument("--campaign-count", type=int, default=25)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--model-name", type=str, default="gpt-4o-mini")
    parser.add_argument("--max-steps", type=int, default=5)
    parser.add_argument("--sample-size", type=int, default=5)
    parser.add_argument(
        "--dataset-path",
        type=str,
        default=None,
        help=f"Optional path to a materialized dataset. Example: {DEFAULT_DATASET_PATH}",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default=None,
        help="Optional path for saving the full experiment artifact as JSON.",
    )
    args = parser.parse_args()

    output = run_experiment(
        campaign_count=args.campaign_count,
        seed=args.seed,
        model_name=args.model_name,
        max_steps=args.max_steps,
        sample_size=args.sample_size,
        dataset_path=args.dataset_path,
    )
    if args.output_path:
        save_experiment_output(output, Path(args.output_path))
    print(json.dumps(output["summary_metrics"], indent=2, sort_keys=True))
    print(json.dumps(output["slice_metrics"], indent=2, sort_keys=True))
    print(json.dumps(output["sample_trajectories"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
