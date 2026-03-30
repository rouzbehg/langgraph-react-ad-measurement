# Synthetic Datasets

This directory is for materialized synthetic campaign datasets generated from the codebase.

These files are different from experiment artifacts:

- dataset files such as `data/synthetic_campaigns.csv`
  - store one row per campaign
  - are used as inputs to the agent and evaluation harness
- experiment artifact files such as `data/experiment_outputs/latest.json`
  - store the outputs of running the agent over a dataset
  - include per-campaign predictions, trajectories, and summary metrics

Example:

```bash
python -m iroas_agent.materialize --count 100 --seed 7 --output data/synthetic_campaigns.csv
```

Supported output formats:

- `.csv`
- `.json`
- `.jsonl`

The exported rows include:

- observed campaign fields visible to the agent
- flattened lift-test and geo-experiment fields
- hidden truth fields for learning and evaluation

Use the runner to create an experiment artifact from a dataset:

```bash
python -m iroas_agent.runner \
  --dataset-path data/synthetic_campaigns.csv \
  --sample-size 20 \
  --output-path data/experiment_outputs/latest.json
```
