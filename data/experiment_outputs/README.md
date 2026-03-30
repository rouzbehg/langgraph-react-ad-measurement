# Experiment Outputs

This directory stores saved experiment artifacts for the notebook-based evals dashboard.

Recommended workflow:

1. Generate or refresh an artifact with the runner:

```bash
source .venv/bin/activate
python -m iroas_agent.runner \
  --dataset-path data/synthetic_campaigns.csv \
  --sample-size 20 \
  --output-path data/experiment_outputs/latest.json
```

2. Open [06_evals_dashboard.ipynb](../../notebooks/06_evals_dashboard.ipynb).

3. Point the notebook at the saved artifact and keep `refresh_experiment = False` for fast reloads.

The saved JSON artifact includes:

- summary metrics
- slice metrics
- full per-campaign results
- sample trajectories
