"""Deterministic config hashing for ML pipeline identity.

Any dataclass that inherits from HashableConfig gets:
  - Stable JSON serialization (sorted keys, no whitespace noise)
  - SHA256 identity hash (same config → same hash, always)
  - Safe canonicalization (rejects non-finite floats, handles numpy/pandas scalars)

Usage:
    from pitight import HashableConfig

    @dataclass(frozen=True)
    class MyModelConfig(HashableConfig):
        learning_rate: float = 0.01
        n_estimators: int = 100

    cfg = MyModelConfig()
    print(cfg.config_hash)  # deterministic SHA256
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from dataclasses import dataclass
from datetime import date, datetime
from decimal import Decimal
from enum import Enum
from pathlib import Path
from typing import Any, Mapping, Self


def canonicalize(x: Any) -> Any:
    """Convert arbitrary Python objects into a JSON-serializable, stable representation.

    This function is intentionally strict — its output feeds identity hashes,
    so silent coercions or ambiguous representations are rejected.

    Supported types:
        None, bool, int, str, float (finite only), Decimal, datetime, date,
        Enum, Path, dataclass, Mapping, list, tuple, set, frozenset,
        and any object with an .item() method (numpy/pandas scalars).

    Raises:
        ValueError: For non-finite floats (NaN, Inf).
        TypeError: For unsupported types or dataclass classes (not instances).
    """
    if x is None or isinstance(x, (bool, int, str)):
        return x

    if isinstance(x, float):
        if x != x or x in (float("inf"), float("-inf")):
            raise ValueError(
                f"Non-finite float ({x!r}) is not allowed in config serialization."
            )
        return x

    if isinstance(x, Decimal):
        return str(x)

    if isinstance(x, datetime):
        return x.isoformat()
    if isinstance(x, date):
        return x.isoformat()

    if isinstance(x, Enum):
        v = x.value
        return v if isinstance(v, (str, int, bool)) else x.name

    if isinstance(x, Path):
        return x.as_posix()

    if isinstance(x, HashableConfig):
        return x.to_json()

    if dataclasses.is_dataclass(x):
        if isinstance(x, type):
            raise TypeError(
                "Dataclass class objects are not supported for config serialization."
            )
        return canonicalize(dataclasses.asdict(x))

    if isinstance(x, Mapping):
        return {str(k): canonicalize(v) for k, v in x.items()}

    if isinstance(x, (list, tuple)):
        return [canonicalize(v) for v in x]

    if isinstance(x, (set, frozenset)):
        items = [canonicalize(v) for v in x]
        try:
            return sorted(items)
        except TypeError:
            return sorted(items, key=lambda v: repr(v))

    # numpy/pandas scalars (best-effort without importing numpy/pandas)
    item = getattr(x, "item", None)
    if callable(item):
        return canonicalize(x.item())

    raise TypeError(f"Unsupported type for config serialization: {type(x).__name__}")


@dataclass(frozen=True, slots=True)
class HashableConfig:
    """Base class for identity-safe configuration objects.

    Guarantees:
      - to_json(): stable, JSON-serializable dict representation
      - to_json_str(): stable JSON string (sorted keys, no whitespace noise)
      - config_hash: stable SHA256 over to_json_str()

    Example:
        @dataclass(frozen=True)
        class TrainConfig(HashableConfig):
            model_type: str = "lightgbm"
            learning_rate: float = 0.05

        cfg = TrainConfig()
        cfg.config_hash  # 'a1b2c3...' (64 hex chars, deterministic)
    """

    def to_json(self) -> dict[str, Any]:
        """Return a stable, JSON-serializable dict of all dataclass fields."""
        raw = {f.name: getattr(self, f.name) for f in dataclasses.fields(self)}
        canon = canonicalize(raw)
        assert isinstance(canon, dict)
        return canon

    def to_json_str(self) -> str:
        """Return a stable JSON string (sorted keys, no whitespace)."""
        return json.dumps(
            self.to_json(),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )

    @property
    def config_hash(self) -> str:
        """SHA256 hex digest of the canonical JSON string."""
        s = self.to_json_str().encode("utf-8")
        return hashlib.sha256(s).hexdigest()

    def short_str(self, hash_len: int = 10, max_len: int = 200) -> str:
        """Human-friendly representation for logs (not for identity)."""
        h = self.config_hash[: max(1, hash_len)]
        body = self.to_json_str()
        if len(body) > max_len:
            body = body[: max_len - 3] + "..."
        return f"{self.__class__.__name__}(hash={h}, json={body})"

    @classmethod
    def from_json(cls, json_str: str) -> Self:
        """Deserialize from a JSON string."""
        data = json.loads(json_str)
        return cls(**data)
