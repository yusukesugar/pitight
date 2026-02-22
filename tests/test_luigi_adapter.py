import pytest

luigi = pytest.importorskip("luigi")

from pitight.artifact import ArtifactRegistry
from pitight.luigi_adapter import artifact_from_luigi_task, register_luigi_output


class DummyTask(luigi.Task):
    ym = luigi.Parameter(default="2025-01")
    version = luigi.Parameter(default="v1")

    def output(self):
        return luigi.LocalTarget(f"/tmp/data/{self.ym}.parquet")


class TestArtifactFromLuigiTask:
    def test_basic(self):
        task = DummyTask(ym="2025-03")
        art = artifact_from_luigi_task(task, config_hash="abc123")
        assert art.name.endswith("DummyTask")
        assert art.path == "/tmp/data/2025-03.parquet"
        assert art.config_hash == "abc123"

    def test_params_in_metadata(self):
        task = DummyTask(ym="2025-03", version="v2")
        art = artifact_from_luigi_task(task, config_hash="abc123")
        assert art.metadata["luigi_params"]["ym"] == "2025-03"
        assert art.metadata["luigi_params"]["version"] == "v2"


class TestRegisterLuigiOutput:
    def test_register(self, tmp_path):
        registry = ArtifactRegistry(str(tmp_path / "artifacts.json"))
        task = DummyTask(ym="2025-06")
        art = register_luigi_output(task, registry, config_hash="def456")
        assert len(registry.list_all()) == 1
        assert art.config_hash == "def456"
