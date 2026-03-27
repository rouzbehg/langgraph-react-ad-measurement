"""Educational LangGraph ReAct system for iROAS estimation."""

from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover
    load_dotenv = None


def _load_project_env() -> None:
    if load_dotenv is None:
        return
    project_root = Path(__file__).resolve().parents[2]
    env_path = project_root / ".env"
    if env_path.exists():
        load_dotenv(env_path, override=False)


_load_project_env()

from .agent import IROASReActAgent
from .data import DEFAULT_DATASET_PATH, generate_campaigns, load_campaign_dataset, write_campaign_dataset
from .runner import run_experiment
from .studio import studio_graph

__all__ = [
    "IROASReActAgent",
    "DEFAULT_DATASET_PATH",
    "generate_campaigns",
    "load_campaign_dataset",
    "write_campaign_dataset",
    "run_experiment",
    "studio_graph",
]
