"""Unit tests for src.core.config.

Covers documented defaults, environment variable overrides, and yaml config
file overrides.
"""

import yaml

from src.core.config import Settings, load_settings


def test_settings_has_documented_defaults():
    settings = Settings()

    assert settings.embedding_model == "models/gemini-embedding-001"
    assert settings.vlm_model == "moondream"
    assert settings.image_dir == "data/images"
    assert settings.top_k == 5
    assert settings.chunk_size == 1000
    assert settings.chunk_overlap == 200
    assert settings.database_url == "postgresql://rag:rag@localhost:5432/rag_multimodal"
    assert settings.redis_url == "redis://localhost:6379/0"
    assert settings.google_api_key is None
    assert settings.openai_api_key is None
    assert settings.allowed_origins == "http://localhost:8501"
    assert settings.log_level == "INFO"
    assert settings.env == "dev"


def test_env_var_override_is_picked_up(monkeypatch):
    monkeypatch.setenv("TOP_K", "9")

    settings = Settings()

    assert settings.top_k == 9


def test_env_var_override_case_insensitive(monkeypatch):
    monkeypatch.setenv("top_k", "9")

    settings = Settings()

    assert settings.top_k == 9


def test_yaml_file_overrides_defaults(tmp_path, monkeypatch):
    configs_dir = tmp_path / "configs"
    configs_dir.mkdir()
    yaml_file = configs_dir / "dev.yml"
    yaml_file.write_text(yaml.dump({"top_k": 8, "vlm_model": "llava"}))

    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("TOP_K", raising=False)

    settings = load_settings()

    assert settings.top_k == 8
    assert settings.vlm_model == "llava"
    # Untouched keys keep their defaults.


def test_env_var_takes_precedence_over_yaml(tmp_path, monkeypatch):
    configs_dir = tmp_path / "configs"
    configs_dir.mkdir()
    yaml_file = configs_dir / "dev.yml"
    yaml_file.write_text(yaml.dump({"top_k": 8}))

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("TOP_K", "20")

    settings = load_settings()

    assert settings.top_k == 20


def test_missing_yaml_file_falls_back_to_defaults(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("TOP_K", raising=False)

    settings = load_settings()

    assert settings.top_k == 5
