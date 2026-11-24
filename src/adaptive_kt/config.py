from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Configuration at {config_path} must contain a YAML mapping.")
    return config


def resolve_path(value: str | Path, root: str | Path | None = None) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute() or root is None:
        return path
    return Path(root).expanduser() / path
