"""Pydantic-settings configuration loaded from env vars and ``.env``."""

from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime settings. Only ``artifacts_dir`` is configurable today."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    artifacts_dir: Path = Path("artifacts")
