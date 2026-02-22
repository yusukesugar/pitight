# pitight

**Pipeline integrity checker for ML workflows.**

Every ML pipeline orchestrator (Airflow, Prefect, Luigi, Dagster) tells you *"your tasks ran successfully."*
None of them tell you *"your tasks ran correctly."*

pitight fills that gap:

- **Config drift detection** — Was this artifact built with the current config, or a stale one?
- **Schema contract validation** — Do your DataFrames match the declared column types and constraints?
- **Artifact registry** — What's been built, what's missing, what's orphaned?
- **Safe pandas operations** — `safe_merge` that rejects silent column collisions
- **Deterministic config hashing** — Same config → same hash, always

## Install

```bash
pip install pitight
```

## Quick Start

### Config Hashing

```python
from dataclasses import dataclass
from pitight import HashableConfig

@dataclass(frozen=True)
class TrainConfig(HashableConfig):
    model_type: str = "lightgbm"
    learning_rate: float = 0.05
    n_estimators: int = 100

cfg = TrainConfig()
print(cfg.config_hash)  # deterministic SHA256, 64 hex chars
# Change any param → different hash → cache invalidation
```

### DataFrame Assertions

```python
import pandas as pd
from pitight import assert_has_columns, assert_unique_key, assert_no_nulls, safe_merge

df = load_my_data()

# Fail fast at I/O boundaries
assert_has_columns(df, ["user_id", "timestamp", "score"], tag="LoadTrainData")
assert_unique_key(df, on=["user_id", "timestamp"], tag="LoadTrainData")
assert_no_nulls(df, ["user_id", "score"], tag="LoadTrainData")

# Merge without silent column shadowing
result = safe_merge(left_df, right_df, on="key")
# If both have a "score" column → ValueError (not "score_x", "score_y")
```

### Artifact Registry

```python
from pitight import Artifact, ArtifactRegistry

registry = ArtifactRegistry(".pitight/artifacts.json")

# Register what you built
registry.register(Artifact(
    name="train_features",
    path="data/features/2025-01.parquet",
    config_hash=cfg.config_hash,
))

# Find problems
registry.find_missing()    # registered but file doesn't exist
registry.find_stale({"train_features": new_cfg.config_hash})  # config changed
registry.find_orphaned({"train_features", "predictions"})     # unknown names
```

### CLI

```bash
pitight status              # summary of registered artifacts
pitight check               # run all checks
pitight check --missing-only
```

## Why?

Built from real pain running an ML pipeline with 80+ Luigi tasks, monthly partitioned data, and walk-forward validation. The same problems hit everyone regardless of orchestrator:

1. "Which config produced this artifact?" → **Config hash tracking**
2. "Did I accidentally use future data?" → **Temporal boundary checks** (coming soon)
3. "Are all my monthly partitions actually there?" → **Artifact registry**
4. "My merge silently created `_x` `_y` columns" → **safe_merge**

## Roadmap

- [ ] Temporal leak detection (decision-time / outcome-time boundary enforcement)
- [ ] Luigi adapter (`Artifact.from_luigi_task()`)
- [ ] Prefect / Dagster adapters
- [ ] Schema metadata auto-generation (`_meta/schema.json`, `stats.json`)
- [ ] `pitight lineage` — dependency tree visualization
- [ ] Web dashboard

## License

MIT
