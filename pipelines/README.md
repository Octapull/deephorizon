# pipelines

Apache Airflow DAGs for data ingest and synthetic data generation.

**Owners:** Stajyer 1 (ingest), Stajyer 2 (synthetic).

## Planned DAGs

| DAG | Schedule | Description |
|:---|:---|:---|
| `eht_ingest` | manual | Download EHT UVFITS, validate with Great Expectations, write to MinIO |
| `synthetic_generation` | manual | Run `eht-imaging` source models, output 128x128 prototyping set |
| `training_data_build` | manual | Build 10K 512x512 clean/degraded pairs, version with DVC |

DAGs go in `dags/`, custom hooks/operators in `plugins/`.
