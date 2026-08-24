from __future__ import annotations

import json
import os
import secrets
from collections.abc import Iterator, Mapping, MutableMapping
from contextlib import contextmanager, suppress
from pathlib import Path

from .filesystem_links import (
    AnchoredDirectory,
    AnchoredDirectoryError,
    AnchoredEntryExists,
    create_entry,
    discard_entry,
    open_child_directory,
    read_entry,
    rename_entry,
    sync_directory,
)

RUNTIME_ENV_KEYS = (
    "LOCAL_LM_CHAT_ENGINE",
    "LOCAL_LM_MEDIA_ENGINE",
    "LOCAL_LM_LLAMA_EXECUTABLE",
    "LOCAL_LM_VLLM_EXECUTABLE",
    "LOCAL_LM_LLAMA_URL",
    "LOCAL_LM_COMFY_EXECUTABLE",
    "LOCAL_LM_COMFY_DIRECTORY",
    "LOCAL_LM_COMFY_URL",
    "LOCAL_LM_WORKER_STARTUP_SECONDS",
)


_STATE_DIR_NAME = "state"
_CONFIG_NAME = "runtime-config.json"
REDIRECTED_STATE = "LM Atelier's state folder may not be a filesystem link"


class RuntimeConfigError(RuntimeError):
    """Runtime configuration could not be read or written safely."""


def runtime_config_path(data_dir: Path) -> Path:
    return data_dir / _STATE_DIR_NAME / _CONFIG_NAME


@contextmanager
def _held_state(data_dir: Path) -> Iterator[AnchoredDirectory]:
    """Hold the state folder for a WHOLE read-modify-write.

    Both callers below read the saved configuration and then write it back.
    Resolving the pathname once for the read and again for the write would be
    two directory identities in one operation, and the file being written here
    feeds process environment on the next launch - so a redirected folder is
    not merely a wrong place to write, it is a way to supply values.

    Establishing the folder is legitimate here because persisting IS the
    requested operation; what must not happen is establishing it through a
    redirection, which acquisition refuses at every component.
    """

    root = (data_dir).expanduser()
    if not root.is_absolute():
        # Acquisition walks from a volume root, so a relative data directory -
        # which the documented .env spelling allows - has to be made absolute
        # first. absolute() prepends the working directory and NOTHING else;
        # resolve() would additionally follow any redirection on the way, which
        # is the step that hides what acquisition exists to refuse.
        root = Path.cwd() / root

    # Every missing component is created THROUGH its held parent, one at a
    # time. A pathname mkdir(parents=True) here created the state folder
    # inside a redirected data root before acquisition ever got to refuse it -
    # creation preceded containment, which is the whole defect in miniature.
    #
    # The chain is retained rather than released as it descends: an adopted
    # child holds only its own handle, so letting go of an ancestor would give
    # back the very thing being held onto.
    held: list[AnchoredDirectory] = []
    try:
        parts = root.parts
        anchor = AnchoredDirectory(Path(parts[0]))
        held.append(anchor)
        for component in (*parts[1:], _STATE_DIR_NAME):
            anchor = open_child_directory(anchor, component, create=True)
            held.append(anchor)
        yield anchor
    except AnchoredDirectoryError as exc:
        raise RuntimeConfigError(REDIRECTED_STATE) from exc
    finally:
        for entry in reversed(held):
            entry.close()


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
    with _held_state(data_dir) as anchor:
        saved = _read_runtime_config(anchor)
        for key, value in saved.items():
            active_environment.setdefault(key, value)

        merged = {
            key: active_environment[key]
            for key in RUNTIME_ENV_KEYS
            if active_environment.get(key, "").strip()
        }
        if explicit or (saved and merged != saved):
            _write_runtime_config(anchor, merged)


def persist_runtime_values(
    data_dir: Path,
    values: Mapping[str, str],
    environment: MutableMapping[str, str] | None = None,
) -> None:
    """Persist an allowlisted runtime update and apply it to this process."""

    active_environment = environment if environment is not None else os.environ
    for key in values:
        if key not in RUNTIME_ENV_KEYS:
            raise ValueError(f"runtime configuration key is not allowed: {key}")
    with _held_state(data_dir) as anchor:
        _persist_held(anchor, values, active_environment)


def _persist_held(
    anchor: AnchoredDirectory,
    values: Mapping[str, str],
    active_environment: MutableMapping[str, str],
) -> None:
    merged = _read_runtime_config(anchor)
    for key, value in values.items():
        cleaned = value.strip()
        if cleaned:
            merged[key] = cleaned
            active_environment[key] = cleaned
        else:
            merged.pop(key, None)
            active_environment.pop(key, None)
    _write_runtime_config(anchor, merged)


def _read_runtime_config(anchor: AnchoredDirectory) -> dict[str, str]:
    """Read the saved configuration through the held state folder."""

    try:
        raw = read_entry(anchor, _CONFIG_NAME)
    except AnchoredDirectoryError as exc:
        raise RuntimeConfigError(REDIRECTED_STATE) from exc
    if raw is None:
        return {}
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, ValueError):
        return {}
    if not isinstance(payload, dict):
        return {}
    return {
        key: value
        for key, value in payload.items()
        if key in RUNTIME_ENV_KEYS and isinstance(value, str) and value.strip()
    }


def _write_runtime_config(anchor: AnchoredDirectory, values: Mapping[str, str]) -> None:
    """Publish the configuration atomically inside the held state folder.

    The staging name is unpredictable and created EXCLUSIVELY. The previous
    fixed ".tmp" sibling could be pre-planted as a link, and a plain write
    follows one - so the guard on the published name protected everything
    except the file the bytes actually travelled through.
    """

    payload = (json.dumps(dict(values), indent=2) + "\n").encode("utf-8")
    # Collision-RESISTANT, not merely per-process. A pid and thread id are
    # deterministic and get reused, so a staging entry left by an interrupted
    # earlier write could block a later one and be misreported as a redirected
    # state folder.
    staging = ""
    descriptor = -1
    for _attempt in range(5):
        staging = f"{_CONFIG_NAME}.{secrets.token_hex(8)}.tmp"
        try:
            descriptor = create_entry(anchor, staging)
            break
        except AnchoredEntryExists:
            continue
        except AnchoredDirectoryError as exc:
            raise RuntimeConfigError(REDIRECTED_STATE) from exc
    if descriptor < 0:
        raise RuntimeConfigError(REDIRECTED_STATE)
    try:
        handle = os.fdopen(descriptor, "wb")
    except BaseException:
        # fdopen takes ownership only once it succeeds, so every failure
        # BEFORE it has to close the descriptor itself.
        with suppress(OSError):
            os.close(descriptor)
        with suppress(Exception):
            discard_entry(anchor, staging)
        raise
    try:
        with handle:
            if hasattr(os, "fchmod"):
                os.fchmod(handle.fileno(), 0o600)
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        with suppress(Exception):
            discard_entry(anchor, staging)
        raise
    try:
        # Publishing over the previous configuration is the whole point, so
        # replacement is requested explicitly.
        rename_entry(anchor, staging, _CONFIG_NAME, replace=True)
        # The record is durable; the directory ENTRY naming it is not until
        # the directory itself is synced.
        sync_directory(anchor)
    except AnchoredDirectoryError as exc:
        with suppress(Exception):
            discard_entry(anchor, staging)
        raise RuntimeConfigError(REDIRECTED_STATE) from exc
