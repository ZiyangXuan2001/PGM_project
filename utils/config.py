"""Configuration loading for controlled DiffTraj-PGM experiments."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_config(path: str) -> dict[str, Any]:
    """Load a YAML config file as a Python dictionary."""

    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"config file not found: {config_path}")
    if not config_path.is_file():
        raise FileNotFoundError(f"config path is not a file: {config_path}")

    try:
        with config_path.open("r", encoding="utf-8") as handle:
            config = yaml.safe_load(handle)
    except yaml.YAMLError as exc:
        raise ValueError(f"malformed YAML config at {config_path}: {exc}") from exc

    if not isinstance(config, dict):
        raise ValueError(f"config at {config_path} must contain a YAML mapping")
    return config

