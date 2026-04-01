from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

DEFAULT_MODEL_NAME = "fastino/gliner2-multi-v1"
DEFAULT_PORT = 8000
DEFAULT_CONFIG_FILE = "config/app_config.json"


@dataclass(frozen=True)
class AppConfig:
    model_name: str
    port: int


def _read_config(path: str) -> dict[str, Any]:
    cfg_path = Path(path)
    if not cfg_path.exists():
        return {}

    raw = json.loads(cfg_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"Archivo de configuracion invalido: {path}")
    return raw


def _parse_port(value: Any) -> int:
    port = int(value)
    if port < 1 or port > 65535:
        raise ValueError(f"Puerto fuera de rango: {port}")
    return port


def get_config() -> AppConfig:
    config_file = os.getenv("APP_CONFIG_FILE", DEFAULT_CONFIG_FILE)
    from_file = _read_config(config_file)

    model_name = (
        os.getenv("APP_MODEL_NAME")
        or os.getenv("MODEL_NAME")
        or str(from_file.get("model_name", DEFAULT_MODEL_NAME))
    )

    port_raw = os.getenv("APP_PORT") or os.getenv("PORT") or from_file.get("port", DEFAULT_PORT)
    port = _parse_port(port_raw)

    return AppConfig(model_name=model_name, port=port)
