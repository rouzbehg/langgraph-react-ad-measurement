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

## Project layout

- `src/iroas_agent/data.py`: synthetic data generation
- `src/iroas_agent/materialize.py`: dataset export utility
- `src/iroas_agent/tools.py`: estimator and diagnostics tools
- `src/iroas_agent/agent.py`: LangGraph ReAct agent
- `src/iroas_agent/evaluation.py`: metric computation and slicing
- `src/iroas_agent/runner.py`: experiment runner
- `prompts/react_agent_prompt.md`: versioned agent prompt
- `notebooks/`: guided learning notebooks

## Notes

- The agent never computes iROAS directly from raw campaign inputs. It must use tools.
- Hidden truth fields exist only for evaluation and tracing metadata.
- LangSmith is treated as the primary debugging interface when enabled.
- Exported datasets in `data/` include both observed fields and hidden truth for learning and evaluation.
