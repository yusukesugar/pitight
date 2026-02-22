"""pitight — Pipeline integrity checker for ML workflows."""

__version__ = "0.4.0"

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
from pitight.temporal_leak import TemporalBoundary, TemporalLeakError, check_leak, no_leak
from pitight.validation import (
    ExecutionError,
    IdentityCalculationError,
    SourceValidationError,
    validate_df_schema,
    validate_source_meta,
)

__all__ = [
    "Artifact",
    "ArtifactRegistry",
    "EmptyDataError",
    "HashableConfig",
    "SchemaViolationError",
    "TemporalBoundary",
    "TemporalLeakError",
    "assert_has_columns",
    "assert_no_nulls",
    "assert_not_empty",
    "assert_unique_key",
    "canonicalize",
    "check_leak",
    "infer_schema",
    "infer_stats",
    "no_leak",
    "safe_merge",
    "validate_df_schema",
    "validate_source_meta",
    "write_schema_and_stats",
    "ExecutionError",
    "IdentityCalculationError",
    "SourceValidationError",
]
