"""Single settings source for rag-multimodal-2026.

Precedence, highest to lowest:
1. Environment variables (case insensitive), including a `.env` file if present.
2. Values loaded from `configs/{env}.yml`, when that file exists.
3. The field defaults declared on the Settings class below.

Rationale: environment variables are the standard way to configure a
twelve factor app and are what deployment tooling (Docker, Kubernetes,
CI) sets at runtime, so they must be able to override anything checked
into a yaml file. The yaml file exists for developer convenience, to
express a full profile (dev, staging, prod) without exporting a pile of
env vars by hand.

Implementation note: pydantic-settings, by default, treats constructor
(init) arguments as higher priority than environment variables. That is
the opposite of the precedence documented above, so `load_settings()`
does not simply pass yaml values in as constructor kwargs. Instead it
resolves the active env name (from the ENV environment variable, case
insensitive, else the class default), reads configs/{env}.yml if it
exists, and only fills in values for fields that were not already set
by a real environment variable (or `.env` file). This keeps env vars
strictly authoritative over yaml, while yaml still overrides the plain
field defaults.
"""

import os
from pathlib import Path
from typing import Any

import yaml
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    database_url: str = "postgresql://rag:rag@localhost:5432/rag_multimodal"
    redis_url: str = "redis://localhost:6379/0"
    google_api_key: str | None = None
    openai_api_key: str | None = None
    embedding_provider: str = "google"  # google or ollama
    embedding_model: str = "models/gemini-embedding-001"
    embedding_dims: int = 768
    ollama_base_url: str = "http://localhost:11434"
    ollama_embedding_model: str = "nomic-embed-text"
    ollama_num_ctx: int = 8192
    # Multimodal controls. Images are captioned at index time by the vision
    # model and stored under image_dir; a single vector search over top_k items
    # returns the most relevant text chunks and image captions together.
    vlm_model: str = "moondream"
    image_dir: str = "data/images"
    top_k: int = 5
    chunk_size: int = 1000
    chunk_overlap: int = 200
    max_upload_mb: int = 25
    allowed_origins: str = "http://localhost:8501"
    log_level: str = "INFO"
    env: str = "dev"


def _env_var_is_set(field_name: str) -> bool:
    """True if an environment variable exists for this field, any case."""
    return field_name.upper() in os.environ or field_name.lower() in os.environ


def _load_yaml_overrides(env_name: str) -> dict[str, Any]:
    """Read configs/{env_name}.yml and return keys matching Settings fields.

    Returns an empty dict if the file does not exist or has no matching
    top level keys.
    """
    yaml_path = Path("configs") / f"{env_name}.yml"
    if not yaml_path.exists():
        return {}

    with yaml_path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    if not isinstance(raw, dict):
        return {}

    valid_fields = set(Settings.model_fields.keys())
    return {key: value for key, value in raw.items() if key in valid_fields}


def load_settings() -> Settings:
    """Construct Settings with env vars taking precedence over yaml.

    Env vars (and `.env`) are read first via a normal Settings()
    construction. For any field that was not actually supplied by an
    environment variable or `.env` file, a matching value from
    configs/{env}.yml (if the file exists) is applied as an override.
    Fields not touched by either source keep their class defaults.
    """
    base = Settings()
    yaml_overrides = _load_yaml_overrides(base.env)

    if not yaml_overrides:
        return base

    applicable = {
        key: value for key, value in yaml_overrides.items() if not _env_var_is_set(key)
    }
    if not applicable:
        return base

    return base.model_copy(update=applicable)


settings = load_settings()
