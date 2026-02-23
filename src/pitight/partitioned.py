"""PartitionedArtifact — declarative lifecycle for a single-partition artifact.

Composes pitight building blocks (assertions, temporal_leak, partition,
schema_stats) into a single ABC that handles:
- Input column validation
- Temporal leak checking
- Output schema enforcement
- Parquet + metadata writing
- Empty data policies

A PartitionedArtifact represents one concrete output (e.g. a feature table)
within a pipeline stage. One stage (package) can contain multiple artifacts:

    features/                   ← stage (package)
    ├── user_engagement         ← PartitionedArtifact
    ├── session_summary         ← PartitionedArtifact
    └── purchase_history        ← PartitionedArtifact

Usage:
    from pitight.partitioned import PartitionedArtifact, InputSpec, EmptyPolicy

    class UserEngagement(PartitionedArtifact):
        artifact_name = "features/user_engagement"
        OUTPUT_SCHEMA = {"user_id": "string", "sessions_7d": "int64"}
        REQUIRED_INPUTS = {
            "events": InputSpec(required_cols=["user_id", "event_type"]),
        }
        EMPTY_POLICY = EmptyPolicy.FAIL

        def compute(self, inputs, period):
            df = inputs["events"]
            return df.groupby("user_id").size().reset_index(name="sessions_7d")
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, ClassVar

import pandas as pd

from pitight.assertions import EmptyDataError, SchemaViolationError, assert_has_columns
from pitight.partition import hive_path
from pitight.schema_stats import write_schema_and_stats
from pitight.temporal_leak import TemporalBoundary, check_leak

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class InputSpec:
    """Specification for a single named input DataFrame."""

    required_cols: list[str] = field(default_factory=list)
    allow_empty: bool = False
    description: str | None = None


class EmptyPolicy(str, Enum):
    """How to handle empty compute output."""

    FAIL = "FAIL"
    WRITE_EMPTY = "WRITE_EMPTY"
    WRITE_EMPTY_IF_UPSTREAM_EMPTY = "WRITE_EMPTY_IF_UPSTREAM_EMPTY"


class PartitionedArtifact(ABC):
    """Abstract base class for a single-partition artifact with declarative lifecycle.

    Represents one concrete output within a pipeline stage.
    ``artifact_name`` may include a stage prefix (e.g. ``"s30_features/binary_454"``).

    Subclasses must define class-level contracts and implement ``compute()``.
    """

    artifact_name: ClassVar[str]
    OUTPUT_SCHEMA: ClassVar[dict[str, str]]
    REQUIRED_INPUTS: ClassVar[dict[str, InputSpec]]
    EMPTY_POLICY: ClassVar[EmptyPolicy] = EmptyPolicy.FAIL

    def __init__(
        self,
        data_root: Path | str,
        *,
        boundary: TemporalBoundary | None = None,
        stage_name: str | None = None,
        freq: str = "M",
    ) -> None:
        self.data_root = Path(data_root)
        self.boundary = boundary
        self.stage_name = stage_name or self.artifact_name
        self.freq = freq

    # ------------------------------------------------------------------
    # Abstract
    # ------------------------------------------------------------------

    @abstractmethod
    def compute(
        self, inputs: dict[str, pd.DataFrame], period: str
    ) -> pd.DataFrame:
        """Transform inputs into output DataFrame for one period."""

    # ------------------------------------------------------------------
    # Optional hooks
    # ------------------------------------------------------------------

    def preprocess_inputs(
        self, inputs: dict[str, pd.DataFrame], period: str
    ) -> dict[str, pd.DataFrame]:
        """Hook: transform inputs before compute. Default: identity."""
        return inputs

    def validate(self, df: pd.DataFrame, period: str) -> None:
        """Hook: custom validation on compute output. Default: no-op."""

    def postprocess(self, df: pd.DataFrame, period: str) -> pd.DataFrame:
        """Hook: transform output after validation. Default: identity."""
        return df

    def build_meta(
        self, df: pd.DataFrame, period: str
    ) -> dict[str, Any] | None:
        """Hook: return extra metadata for stats.json. Default: None."""
        return None

    # ------------------------------------------------------------------
    # Completion check
    # ------------------------------------------------------------------

    def output_path(self, period: str) -> Path:
        """Return the expected parquet path for a period."""
        return hive_path(
            self.data_root / self.artifact_name,
            period,
            freq=self.freq,
        )

    def is_complete(self, period: str) -> bool:
        """Check if output parquet exists for a period.

        Data-presence-based completion: independent of who requested it
        or what period range the orchestrator used. Solves the Luigi
        run_id mismatch problem where data exists but manifest doesn't.
        """
        return self.output_path(period).exists()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def run(
        self,
        period: str,
        inputs: dict[str, pd.DataFrame],
        *,
        upstream_empty: bool = False,
    ) -> Path:
        """Execute the full artifact lifecycle for one period.

        Args:
            period: Partition period string (e.g. "2025-01").
            inputs: Named input DataFrames matching REQUIRED_INPUTS keys.
            upstream_empty: Whether upstream produced empty data.

        Returns:
            Path to the written parquet file.

        Raises:
            SchemaViolationError: On input/output schema violations.
            EmptyDataError: On empty data when policy forbids it.
            TemporalLeakError: On temporal leak detection.
        """
        tag = f"{self.stage_name}[{period}]"

        # 1. Validate input keys and columns
        self._validate_inputs(inputs, tag)

        # 2. preprocess_inputs hook
        inputs = self.preprocess_inputs(inputs, period)

        # 3. compute
        df = self.compute(inputs, period)

        # 4. Temporal leak check
        if self.boundary is not None:
            check_leak(df, self.boundary, self.stage_name, raise_on_violation=True)

        # 5. validate hook
        self.validate(df, period)

        # 6. postprocess hook
        df = self.postprocess(df, period)

        # 7. Schema enforcement — missing columns + column ordering
        self._enforce_output_schema(df, tag)
        df = df[list(self.OUTPUT_SCHEMA.keys())]

        # 8. Empty policy
        if len(df) == 0:
            self._handle_empty(upstream_empty, tag)

        # 9. Write parquet + schema.json + stats.json
        out_path = self.output_path(period)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(out_path, index=False)
        logger.info("%s: wrote %d rows → %s", tag, len(df), out_path)

        meta_dir = out_path.parent / "_meta"
        extra = self.build_meta(df, period)
        write_schema_and_stats(
            df,
            meta_dir,
            out_path.stem,
            extra_stats=extra,
        )

        return out_path

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _validate_inputs(
        self, inputs: dict[str, pd.DataFrame], tag: str
    ) -> None:
        """Check that all required input keys exist and columns are present."""
        missing_keys = set(self.REQUIRED_INPUTS) - set(inputs)
        if missing_keys:
            raise SchemaViolationError(
                f"{tag}: missing input keys: {sorted(missing_keys)}"
            )

        for name, spec in self.REQUIRED_INPUTS.items():
            df = inputs[name]
            if not spec.allow_empty and len(df) == 0:
                raise EmptyDataError(
                    f"{tag}: input '{name}' is empty (allow_empty=False)"
                )
            if spec.required_cols:
                assert_has_columns(df, spec.required_cols, tag=f"{tag}.{name}")

    def _enforce_output_schema(self, df: pd.DataFrame, tag: str) -> None:
        """Check that all OUTPUT_SCHEMA columns exist in df."""
        expected = set(self.OUTPUT_SCHEMA)
        actual = set(df.columns)
        missing = expected - actual
        if missing:
            raise SchemaViolationError(
                f"{tag}: output missing columns: {sorted(missing)}"
            )

    def _handle_empty(self, upstream_empty: bool, tag: str) -> None:
        """Apply empty policy. Raises EmptyDataError when policy says FAIL."""
        if self.EMPTY_POLICY == EmptyPolicy.FAIL:
            raise EmptyDataError(f"{tag}: compute returned empty DataFrame")
        if self.EMPTY_POLICY == EmptyPolicy.WRITE_EMPTY:
            logger.warning("%s: writing empty parquet (WRITE_EMPTY policy)", tag)
            return
        # WRITE_EMPTY_IF_UPSTREAM_EMPTY
        if upstream_empty:
            logger.warning(
                "%s: writing empty parquet (upstream_empty=True)", tag
            )
            return
        raise EmptyDataError(
            f"{tag}: compute returned empty DataFrame "
            f"(upstream_empty=False, policy=WRITE_EMPTY_IF_UPSTREAM_EMPTY)"
        )


# Backward compatibility alias
Stage = PartitionedArtifact
