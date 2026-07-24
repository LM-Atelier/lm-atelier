from __future__ import annotations

import json
import os
from collections.abc import Mapping, MutableMapping
from contextlib import suppress
from pathlib import Path

RUNTIME_ENV_KEYS = (
    "LOCAL_LM_CHAT_ENGINE",
    "LOCAL_LM_MEDIA_ENGINE",
    "LOCAL_LM_LLAMA_EXECUTABLE",
    "LOCAL_LM_LLAMA_URL",
    "LOCAL_LM_COMFY_EXECUTABLE",
    "LOCAL_LM_COMFY_DIRECTORY",
    "LOCAL_LM_COMFY_URL",
)


def runtime_config_path(data_dir: Path) -> Path:
    return data_dir / "state" / "runtime-config.json"


def configure_persisted_runtime(
    data_dir: Path,
    environment: MutableMapping[str, str] | None = None,
) -> None:
    """Restore runtime paths and retain explicit overrides for future launches.

    Only the allowlisted, non-secret engine configuration is persisted. Explicit
    environment variables always take precedence over values saved previously.
    """

    active_environment = environment if environment is not None else os.environ
    explicit = {
        key: active_environment[key]
        for key in RUNTIME_ENV_KEYS
        if active_environment.get(key, "").strip()
    }
    saved = _read_runtime_config(runtime_config_path(data_dir))
    for key, value in saved.items():
        active_environment.setdefault(key, value)

    merged = {
        key: active_environment[key]
        for key in RUNTIME_ENV_KEYS
        if active_environment.get(key, "").strip()
    }
    if explicit or (saved and merged != saved):
        _write_runtime_config(runtime_config_path(data_dir), merged)


def _read_runtime_config(path: Path) -> dict[str, str]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, ValueError):
        return {}
    if not isinstance(payload, dict):
        return {}
    return {
        key: value
        for key, value in payload.items()
        if key in RUNTIME_ENV_KEYS and isinstance(value, str) and value.strip()
    }


def _write_runtime_config(path: Path, values: Mapping[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(dict(values), indent=2) + "\n", encoding="utf-8")
    with suppress(OSError):
        temporary.chmod(0o600)
    os.replace(temporary, path)
    with suppress(OSError):
        path.chmod(0o600)
