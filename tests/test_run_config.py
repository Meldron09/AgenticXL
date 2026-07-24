from __future__ import annotations

from agenticxl.config import RunConfig, _parse_bool_env

MINIMAL_YAML = """
input_dir: Documents
keywords: [aerial lift]
enrichment_columns: [Category]
"""


def _write_config(tmp_path):
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    path = config_dir / "run_config.yaml"
    path.write_text(MINIMAL_YAML)
    return path


def test_skip_enrichment_defaults_to_false(tmp_path):
    config_path = _write_config(tmp_path)
    run_config = RunConfig.load(config_path, project_root=tmp_path, env={"ANTHROPIC_API_KEY": "sk-x"})
    assert run_config.skip_enrichment is False


def test_skip_enrichment_true_from_env(tmp_path):
    config_path = _write_config(tmp_path)
    run_config = RunConfig.load(
        config_path,
        project_root=tmp_path,
        env={"ANTHROPIC_API_KEY": "sk-x", "AGENTICXL_SKIP_ENRICHMENT": "true"},
    )
    assert run_config.skip_enrichment is True


def test_skip_enrichment_accepts_common_truthy_spellings():
    for value in ("1", "true", "True", "YES", "on"):
        assert _parse_bool_env(value) is True
    for value in (None, "", "0", "false", "no", "off", "garbage"):
        assert _parse_bool_env(value) is False
