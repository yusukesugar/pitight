"""pitight — Pipeline integrity checker for ML workflows."""

__version__ = "0.2.0"

from pitight.artifact import Artifact, ArtifactRegistry
from pitight.assertions import (
    EmptyDataError,
    SchemaViolationError,
    assert_has_columns,
    assert_no_nulls,
    assert_not_empty,
    assert_unique_key,
    safe_merge,
)
from pitight.config_hash import HashableConfig, canonicalize
from pitight.schema_stats import infer_schema, infer_stats, write_schema_and_stats

__all__ = [
    "Artifact",
    "ArtifactRegistry",
    "EmptyDataError",
    "HashableConfig",
    "SchemaViolationError",
    "assert_has_columns",
    "assert_no_nulls",
    "assert_not_empty",
    "assert_unique_key",
    "canonicalize",
    "infer_schema",
    "infer_stats",
    "safe_merge",
    "write_schema_and_stats",
]
