# Synthetic Datasets

This directory is for materialized synthetic campaign datasets generated from the codebase.

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
