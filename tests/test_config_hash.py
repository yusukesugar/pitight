from dataclasses import dataclass
from decimal import Decimal
from enum import Enum
from pathlib import Path

import pytest

from pitight.config_hash import HashableConfig, canonicalize


@dataclass(frozen=True)
class SimpleConfig(HashableConfig):
    lr: float = 0.01
    n_trees: int = 100
    name: str = "lgbm"


@dataclass(frozen=True)
class NestedConfig(HashableConfig):
    base: SimpleConfig = SimpleConfig()
    extra: str = "hello"


class Color(Enum):
    RED = "red"
    GREEN = "green"


class TestCanonicalize:
    def test_primitives(self):
        assert canonicalize(None) is None
        assert canonicalize(True) is True
        assert canonicalize(42) == 42
        assert canonicalize("hello") == "hello"
        assert canonicalize(3.14) == 3.14

    def test_nan_rejected(self):
        with pytest.raises(ValueError, match="Non-finite"):
            canonicalize(float("nan"))

    def test_inf_rejected(self):
        with pytest.raises(ValueError, match="Non-finite"):
            canonicalize(float("inf"))

    def test_decimal(self):
        assert canonicalize(Decimal("1.23")) == "1.23"

    def test_enum(self):
        assert canonicalize(Color.RED) == "red"

    def test_path(self):
        assert canonicalize(Path("/tmp/data")) == "/tmp/data"

    def test_set_sorted(self):
        result = canonicalize({3, 1, 2})
        assert result == [1, 2, 3]

    def test_unsupported_type(self):
        with pytest.raises(TypeError, match="Unsupported"):
            canonicalize(object())


class TestHashableConfig:
    def test_deterministic_hash(self):
        c1 = SimpleConfig()
        c2 = SimpleConfig(lr=0.01, n_trees=100, name="lgbm")
        assert c1.config_hash == c2.config_hash

    def test_different_params_different_hash(self):
        c1 = SimpleConfig(lr=0.01)
        c2 = SimpleConfig(lr=0.02)
        assert c1.config_hash != c2.config_hash

    def test_to_json(self):
        cfg = SimpleConfig()
        j = cfg.to_json()
        assert j == {"lr": 0.01, "n_trees": 100, "name": "lgbm"}

    def test_to_json_str_sorted(self):
        cfg = SimpleConfig()
        s = cfg.to_json_str()
        assert '"lr"' in s
        # Keys are sorted
        assert s.index('"lr"') < s.index('"n_trees"')

    def test_nested_config(self):
        cfg = NestedConfig()
        j = cfg.to_json()
        assert j["base"] == {"lr": 0.01, "n_trees": 100, "name": "lgbm"}
        assert isinstance(cfg.config_hash, str)
        assert len(cfg.config_hash) == 64  # SHA256 hex

    def test_from_json_roundtrip(self):
        cfg = SimpleConfig(lr=0.05, n_trees=200, name="xgb")
        restored = SimpleConfig.from_json(cfg.to_json_str())
        assert restored == cfg
        assert restored.config_hash == cfg.config_hash

    def test_short_str(self):
        cfg = SimpleConfig()
        s = cfg.short_str()
        assert "SimpleConfig" in s
        assert "hash=" in s
