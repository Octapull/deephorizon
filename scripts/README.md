# scripts

Standalone CLI scripts for data acquisition, synthetic data generation, and visualization. Each script is **runnable from the terminal** for fast iteration and is also **invoked from Airflow DAGs** (see `pipelines/dags/`) for scheduled pipeline runs.

**Owners:** Stajyer 1 (EHT ingest), Stajyer 2 (synthetic generation).

## Scripts

| Script | Owner | Purpose |
|:---|:---|:---|
| `download_eht_data.py` | Stajyer 1 | Download 88 EHT UVFITS files (7 datasets) |
| `generate_synthetic_data.py` | Stajyer 2 | 128x128 prototyping pairs (`eht-imaging`) |
| `generate_training_data.py` | Stajyer 2 | 10,000 512x512 clean/degraded pairs for training |
| `visualize_samples.py` | Stajyer 1/2 | Render dirty images and PNG comparisons |
| `eval_baseline.py` | Stajyer 5 (Week 3) | No-ML baseline metrics (bicubic upsample) |

## DRY rule

Each script should expose a `main()` function with `argparse`-driven flags. Airflow DAGs in `pipelines/dags/` must call the script (`BashOperator` or `PythonOperator` importing `main`) — **never reimplement the logic in the DAG body**. The script is the source of truth; the DAG just schedules it.

## Install

Run with the `data` extras venv:

```bash
uv sync --extra data
uv run python scripts/download_eht_data.py --help
```
