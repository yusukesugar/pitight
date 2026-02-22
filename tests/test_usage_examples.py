"""Usage examples as tests — if these pass, the README examples work.

Each test class demonstrates a real-world use case. New users should
start here to understand how pitight works.
"""

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import pytest

from pitight import (
    Artifact,
    ArtifactRegistry,
    HashableConfig,
    SchemaViolationError,
    assert_has_columns,
    assert_no_nulls,
    assert_unique_key,
    safe_merge,
)
from pitight.temporal_leak import TemporalBoundary, TemporalLeakError, check_leak, no_leak


# ============================================================
# 1. Config Hashing — "same config = same hash, always"
# ============================================================


class TestConfigHashingWorkflow:
    """You have ML model configs. You need deterministic identity."""

    def test_basic_config(self):
        """Define a config, get a stable hash."""

        @dataclass(frozen=True)
        class TrainConfig(HashableConfig):
            model_type: str = "lightgbm"
            learning_rate: float = 0.05
            n_estimators: int = 100

        cfg = TrainConfig()

        # Hash is deterministic — same config always produces same hash
        assert cfg.config_hash == TrainConfig().config_hash
        assert len(cfg.config_hash) == 64  # SHA256 hex digest

    def test_config_change_invalidates_cache(self):
        """When you change a parameter, the hash changes → cache invalidation."""

        @dataclass(frozen=True)
        class TrainConfig(HashableConfig):
            learning_rate: float = 0.05
            n_estimators: int = 100

        v1 = TrainConfig(learning_rate=0.05)
        v2 = TrainConfig(learning_rate=0.01)  # changed!

        assert v1.config_hash != v2.config_hash

    def test_nested_configs(self):
        """Configs can nest. The parent hash changes when any child changes."""

        @dataclass(frozen=True)
        class FeatureConfig(HashableConfig):
            window_size: int = 20

        @dataclass(frozen=True)
        class PipelineConfig(HashableConfig):
            features: FeatureConfig = FeatureConfig()
            model_type: str = "lgbm"

        cfg = PipelineConfig()
        # Nested config is included in the hash
        assert "window_size" in cfg.to_json_str()

        # Changing nested config changes parent hash
        cfg2 = PipelineConfig(features=FeatureConfig(window_size=50))
        assert cfg.config_hash != cfg2.config_hash

    def test_roundtrip_serialization(self):
        """Configs can be serialized to JSON and back."""

        @dataclass(frozen=True)
        class ModelConfig(HashableConfig):
            name: str = "xgboost"
            depth: int = 6

        original = ModelConfig(name="lightgbm", depth=8)
        restored = ModelConfig.from_json(original.to_json_str())

        assert restored == original
        assert restored.config_hash == original.config_hash


# ============================================================
# 2. DataFrame Assertions — "fail fast at I/O boundaries"
# ============================================================


class TestDataFrameAssertions:
    """You load data from upstream. Validate it before processing."""

    def test_validate_input_dataframe(self):
        """Check that required columns exist and keys are unique."""
        df = pd.DataFrame(
            {
                "user_id": [1, 2, 3],
                "timestamp": ["2025-01-01", "2025-01-02", "2025-01-03"],
                "score": [0.8, 0.6, 0.9],
            }
        )

        # These pass silently — your data is clean
        assert_has_columns(df, ["user_id", "timestamp", "score"], tag="LoadTrainData")
        assert_unique_key(df, on=["user_id", "timestamp"], tag="LoadTrainData")
        assert_no_nulls(df, ["user_id", "score"], tag="LoadTrainData")

    def test_missing_columns_caught_immediately(self):
        """If upstream schema changed, you find out right away."""
        df = pd.DataFrame({"user_id": [1], "value": [100]})

        with pytest.raises(SchemaViolationError, match="missing columns"):
            assert_has_columns(df, ["user_id", "score"], tag="LoadTrainData")

    def test_safe_merge_prevents_silent_column_collision(self):
        """pd.merge silently creates 'col_x', 'col_y'. safe_merge raises instead."""
        users = pd.DataFrame({"id": [1, 2], "name": ["Alice", "Bob"]})
        scores = pd.DataFrame({"id": [1, 2], "value": [100, 200]})

        # No collision — works fine
        result = safe_merge(users, scores, on="id")
        assert list(result.columns) == ["id", "name", "value"]

        # Collision — raises ValueError instead of creating name_x, name_y
        users2 = pd.DataFrame({"id": [1], "name": ["Alice"]})
        scores2 = pd.DataFrame({"id": [1], "name": ["Bob"]})
        with pytest.raises(ValueError):
            safe_merge(users2, scores2, on="id")


# ============================================================
# 3. Artifact Registry — "what's built, what's stale, what's missing"
# ============================================================


class TestArtifactRegistryWorkflow:
    """You have 50+ pipeline outputs. Track what exists and what needs rebuilding."""

    def test_register_and_check_staleness(self, tmp_path):
        """Register artifacts, then detect when configs change."""
        registry = ArtifactRegistry(str(tmp_path / "artifacts.json"))

        # After your pipeline runs, register outputs
        data_file = tmp_path / "features.parquet"
        data_file.write_text("fake parquet data")

        registry.register(
            Artifact(
                name="train_features",
                path=str(data_file),
                config_hash="aaa111",
            )
        )

        # Later, you update your config
        new_configs = {"train_features": "bbb222"}  # hash changed!
        stale = registry.find_stale(new_configs)

        assert len(stale) == 1
        assert stale[0].name == "train_features"
        # → You know this artifact needs rebuilding

    def test_detect_missing_files(self, tmp_path):
        """Find artifacts that are registered but whose files were deleted."""
        registry = ArtifactRegistry(str(tmp_path / "artifacts.json"))

        registry.register(
            Artifact(
                name="predictions",
                path="/nonexistent/predictions.parquet",
                config_hash="abc",
            )
        )

        missing = registry.find_missing()
        assert len(missing) == 1
        assert missing[0].name == "predictions"

    def test_detect_orphaned_artifacts(self, tmp_path):
        """Find artifacts whose stage was removed from the pipeline."""
        registry = ArtifactRegistry(str(tmp_path / "artifacts.json"))

        registry.register(
            Artifact(name="old_experiment", path="/tmp/old.parquet", config_hash="abc")
        )

        # Your current pipeline only has these stages
        current_stages = {"train_features", "predictions"}
        orphaned = registry.find_orphaned(current_stages)

        assert len(orphaned) == 1
        assert orphaned[0].name == "old_experiment"


# ============================================================
# 4. Temporal Leak Detection — "no future data in training"
# ============================================================


class TestTemporalLeakWorkflow:
    """You have a prediction pipeline. Outcome data must not leak into
    decision-time stages."""

    def test_define_and_enforce_boundary(self):
        """Define which columns are forbidden at which stages."""
        boundary = TemporalBoundary(
            forbidden_columns=frozenset({"target", "actual_revenue"}),
            forbidden_prefixes=("outcome_",),
            allowed_stages=frozenset({"evaluation"}),
        )

        # Clean feature data — passes
        features = pd.DataFrame({"user_id": [1], "feature_a": [0.5]})
        violations = check_leak(features, boundary, stage="training")
        assert violations == []

        # Leaked data — caught!
        leaked = pd.DataFrame({"user_id": [1], "feature_a": [0.5], "target": [1]})
        with pytest.raises(TemporalLeakError, match="target"):
            check_leak(leaked, boundary, stage="training")

        # Same columns in evaluation stage — allowed
        violations = check_leak(leaked, boundary, stage="evaluation")
        assert violations == []

    def test_no_leak_decorator(self):
        """Wrap your compute functions with automatic leak checking."""
        boundary = TemporalBoundary(
            forbidden_columns=frozenset({"price_tomorrow"}),
        )

        @no_leak(boundary, stage="feature_engineering")
        def compute_features(raw_df: pd.DataFrame) -> pd.DataFrame:
            return raw_df[["date", "volume"]]

        # Clean input — works
        clean = pd.DataFrame({"date": ["2025-01-01"], "volume": [1000]})
        result = compute_features(clean)
        assert len(result) == 1

        # Leaked input — blocked before your function even runs
        dirty = pd.DataFrame(
            {"date": ["2025-01-01"], "volume": [1000], "price_tomorrow": [150.0]}
        )
        with pytest.raises(TemporalLeakError):
            compute_features(dirty)
