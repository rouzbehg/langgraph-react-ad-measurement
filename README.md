# LangGraph ReAct iROAS Agent

This repository contains a minimal but extensible educational project for studying how a LangGraph-based ReAct agent selects among causal measurement estimators under partial data availability.

## What is included

- Synthetic campaign generator with hidden ground truth
- Estimator tools for RCT, geo diff-in-diff, observational fallback, and diagnostics
- Prompt-file-backed ReAct loop implemented with LangGraph
- LangSmith tracing hooks for full trajectories and tool spans
- Experiment runner and evaluation harness
- Jupyter notebooks for learning and inspection

## Quick start

1. Create a virtual environment with Python 3.9+.
2. Install dependencies:

```bash
pip install -e .
```

3. Set environment variables as needed:

```bash
export OPENAI_API_KEY=...
export LANGSMITH_API_KEY=...
export LANGSMITH_TRACING=true
export LANGSMITH_PROJECT=langgraph-react-ad-measurement
```

4. Run a small experiment:

```bash
python -m iroas_agent.runner --campaign-count 10 --sample-size 3
```

5. Materialize a reproducible synthetic dataset to disk:

```bash
python -m iroas_agent.materialize --count 100 --seed 7 --output data/synthetic_campaigns.csv
```

6. Run the experiment against the materialized dataset:

```bash
python -m iroas_agent.runner --dataset-path data/synthetic_campaigns.csv --sample-size 3
```

7. Save a reusable experiment artifact for the notebook dashboard:

```bash
python -m iroas_agent.runner \
  --dataset-path data/synthetic_campaigns.csv \
  --sample-size 20 \
  --output-path data/experiment_outputs/latest.json
```

8. Launch the local browser dashboard:

```bash
streamlit run apps/evals_dashboard.py
```

The local app reads saved experiment artifacts and, if needed, can regenerate them through the same artifact-first workflow used by the notebook dashboard.

## Data vs artifacts

This project has two different output concepts:

- `data/synthetic_campaigns.csv`
  - a materialized synthetic dataset
  - one row per campaign
  - includes observed campaign fields plus hidden truth for evaluation
- `data/experiment_outputs/latest.json`
  - a saved experiment artifact
  - includes full agent-run outputs such as summary metrics, per-campaign predictions, trajectories, and sample traces

The notebook dashboard is designed to read experiment artifacts, not raw datasets. The raw dataset provides campaign inputs and ground truth; the experiment artifact provides agent behavior and evaluation outputs.

You may also see notebook-local artifacts under `notebooks/data/experiment_outputs/` if a notebook was run from inside the `notebooks/` directory. Those are local convenience outputs and are ignored by Git.

## LangSmith Studio

This repository now includes a LangSmith Studio / LangGraph local server entrypoint via [langgraph.json](/Users/rouzbehgerami/Library/CloudStorage/GoogleDrive-rouzbehg@gmail.com/My%20Drive/Documents/Projects/LLMs/langgraph-react-ad-measurement/langgraph.json) and [studio.py](/Users/rouzbehgerami/Library/CloudStorage/GoogleDrive-rouzbehg@gmail.com/My%20Drive/Documents/Projects/LLMs/langgraph-react-ad-measurement/src/iroas_agent/studio.py).

Official docs note that the LangGraph CLI local server requires Python 3.11+ for Studio usage. This project code has otherwise been developed in Python 3.9, so the cleanest Studio workflow is to use a Python 3.11+ environment for the local server.

Typical setup:

```bash
pip install -U "langgraph-cli[inmem]"
langgraph dev
```

With the server running, Studio should be available at a URL like:

```text
https://smith.langchain.com/studio/?baseUrl=http://127.0.0.1:2024
```

Studio input shape:

```json
{
  "campaign_id": "campaign_0000",
  "dataset_path": "data/synthetic_campaigns.csv"
}
```

If `campaign_id` is omitted, the Studio graph will use the first campaign in the dataset.

## Project layout

- `src/iroas_agent/data.py`: synthetic data generation
- `src/iroas_agent/materialize.py`: dataset export utility
- `src/iroas_agent/dashboard.py`: notebook dashboard helpers and artifact loading
- `apps/evals_dashboard.py`: local browser dashboard built with Streamlit
- `src/iroas_agent/tools.py`: estimator and diagnostics tools
- `src/iroas_agent/agent.py`: LangGraph ReAct agent
- `src/iroas_agent/studio.py`: Studio-friendly graph entrypoint
- `src/iroas_agent/evaluation.py`: metric computation and slicing
- `src/iroas_agent/runner.py`: experiment runner
- `langgraph.json`: LangGraph local server configuration
- `prompts/react_agent_prompt.md`: versioned agent prompt
- `notebooks/`: guided learning notebooks

## Notes

- The agent never computes iROAS directly from raw campaign inputs. It must use tools.
- Hidden truth fields exist only for evaluation and tracing metadata.
- LangSmith is treated as the primary debugging interface when enabled.
- Exported datasets in `data/` include both observed fields and hidden truth for learning and evaluation.
- In the dashboard, `final_estimator_used` means the estimator whose numeric output matches the final predicted answer. This is different from `tool_sequence`, which records every estimator the agent called during comparison.
