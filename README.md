# pitight

**The integrity layer for ML pipelines.**

Every ML pipeline orchestrator (Airflow, Prefect, Luigi, Dagster) tells you *"your tasks ran successfully."*
None of them tell you *"your tasks ran correctly."*

pitight fills that gap — orchestrator-agnostic validation, schema enforcement, and artifact tracking for ML workflows.

## Vision

pitight aims to be **FastAPI for ML pipeline integrity**: declare your schemas and boundaries, and everything else — validation, metadata, leak detection, artifact tracking — is handled automatically.

```python
# Where we're heading:
@pipeline.stage("s30_features", depends=["s20_ratings"])
class TrainFeatures(Stage):
    INPUT = Schema(user_id=int, date=str)
    OUTPUT = Schema(user_id=int, feature_a=float, nullable={"feature_a": False})

    def compute(self, df):
        return transform(df)

# One declaration gives you:
#   Schema validation (input & output)
#   Temporal leak detection
#   Hive-partitioned storage
#   Metadata auto-generation (schema.json, stats.json, manifest.json)
#   Config-driven cache invalidation
#   Coverage tracking & artifact registry
```

Today, pitight provides the building blocks. The declarative `Pipeline` / `Stage` API is coming.

## Install

```bash
pip install pitight                        # from PyPI (planned)
pip install git+https://github.com/yusukesugar/pitight.git  # from GitHub
```

## What's included (v0.5)

### Config Hashing — *"same config = same hash, always"*

```python
from dataclasses import dataclass
from pitight import HashableConfig

@dataclass(frozen=True)
class TrainConfig(HashableConfig):
    model_type: str = "lightgbm"
    learning_rate: float = 0.05

cfg = TrainConfig()
print(cfg.config_hash)  # deterministic SHA256, 64 hex chars

# Change any param -> different hash -> automatic cache invalidation
cfg2 = TrainConfig(learning_rate=0.01)
assert cfg.config_hash != cfg2.config_hash
```

### DataFrame Assertions — *"fail fast at I/O boundaries"*

```python
from pitight import assert_has_columns, assert_unique_key, assert_no_nulls, safe_merge

df = load_upstream_data()
assert_has_columns(df, ["user_id", "timestamp", "score"], tag="LoadTrain")
assert_unique_key(df, on=["user_id", "timestamp"], tag="LoadTrain")
assert_no_nulls(df, ["user_id", "score"], tag="LoadTrain")

# safe_merge: rejects silent column collisions (no more _x, _y surprises)
result = safe_merge(left, right, on="key")
```

### Temporal Leak Detection — *"no future data in training"*

```python
from pitight import TemporalBoundary, check_leak

boundary = TemporalBoundary(
    forbidden_columns=frozenset({"target", "actual_revenue"}),
    forbidden_prefixes=("outcome_",),
    allowed_stages=frozenset({"evaluation"}),
)

# In your feature engineering stage:
check_leak(df, boundary, stage="training")    # raises if "target" is present
check_leak(df, boundary, stage="evaluation")  # allowed here
```

### Partition Management — *"monthly, daily, any frequency"*

```python
from pitight import resolve_expected_periods, compute_coverage, hive_path, scan_present_months

# Enumerate expected periods
periods = resolve_expected_periods("2025-01", "2025-06", freq="M")
# ["2025-01", "2025-02", ..., "2025-06"]

# Check what's actually there
present = scan_present_months(data_dir)
missing, ok = compute_coverage(periods, present)

# Generate hive-style paths
path = hive_path(root, "2025-03", freq="M")
# root/data/year=2025/month=03/part-2025-03.parquet
```

### Artifact Registry — *"what's built, what's stale, what's missing"*

```python
from pitight import Artifact, ArtifactRegistry

registry = ArtifactRegistry(".pitight/artifacts.json")
registry.register(Artifact(name="features", path="data/features.parquet", config_hash=cfg.config_hash))

registry.find_missing()     # file doesn't exist on disk
registry.find_stale({"features": new_hash})  # config changed
registry.find_orphaned({"features", "predictions"})  # unknown names
```

### Schema & Stats — *"auto-generated metadata"*

```python
from pitight import write_schema_and_stats

write_schema_and_stats(df, out_dir=Path("data/_meta"), base_name="features")
# Creates: features.schema.json + features.stats.json
```

### Validation — *"does the data match the contract?"*

```python
from pitight import validate_source_meta, validate_df_schema

# Check that metadata files exist and dates are consistent
validate_source_meta(Path("data/features/"), start_ym="2025-01", end_ym="2025-06")

# Validate DataFrame against its schema.json
validate_df_schema(df, Path("data/_meta/features.schema.json"))
```

## Architecture

```
pitight (building blocks — available now)
    config_hash       Deterministic config hashing (SHA256)
    assertions        DataFrame validation (columns, uniqueness, nulls, safe_merge)
    temporal_leak     Temporal boundary enforcement (column-level)
    partition         Time-partitioned storage (hive paths, coverage, manifest)
    artifact          Artifact registry (JSON-backed tracking)
    schema_stats      Metadata generation (schema.json, stats.json)
    validation        Schema & metadata file validation

pitight "FastAPI layer" (coming)
    Pipeline          Declarative pipeline definition
    Stage             Stage with auto-validation, auto-metadata, auto-storage
    adapters/         Thin wrappers for Luigi, Airflow, Prefect, Dagster
    dashboard/        Web UI for pipeline health
```

## Design Principles

1. **Orchestrator-agnostic** — pitight is a checker, not a scheduler. Works with any orchestrator or none.
2. **Declare, don't code** — Define schemas and boundaries; pitight handles the rest.
3. **Fail fast** — Catch problems at I/O boundaries, not 3 stages downstream.
4. **Zero magic** — Every function is importable and testable independently.
5. **Incremental adoption** — Use one function or the whole framework. No lock-in.

## Built from production

Extracted from a production ML pipeline (80+ tasks, 18-month walk-forward validation, monthly partitioned data). Every feature exists because something broke in production first.

## License

MIT
