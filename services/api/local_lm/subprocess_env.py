from __future__ import annotations

import os
from collections.abc import Mapping

# Subprocesses receive only operating-system/runtime context that is required to
# start reliably. Credentials and application configuration are passed through
# explicit channels instead of inheriting the API process environment.
_PASSTHROUGH_KEYS = (
    "ALLUSERSPROFILE",
    "APPDATA",
    "COMSPEC",
    "CommonProgramFiles",
    "CommonProgramFiles(x86)",
    "CommonProgramW6432",
    "HOMEDRIVE",
    "HOMEPATH",
    "HOME",
    "LANG",
    "LC_ALL",
    "LC_CTYPE",
    "LOCALAPPDATA",
    "PATH",
    "PATHEXT",
    "PROGRAMDATA",
    "ProgramFiles",
    "ProgramFiles(x86)",
    "ProgramW6432",
    "SHELL",
    "SYSTEMDRIVE",
    "SYSTEMROOT",
    "TEMP",
    "TMP",
    "TMPDIR",
    "USERPROFILE",
    "WINDIR",
)

_ACCELERATOR_KEYS = (
    "CUBLAS_WORKSPACE_CONFIG",
    "CUDA_CACHE_PATH",
    "CUDA_HOME",
    "CUDA_MODULE_LOADING",
    "CUDA_PATH",
    "CUDA_VISIBLE_DEVICES",
    "GPU_DEVICE_ORDINAL",
    "HIP_PATH",
    "HIP_VISIBLE_DEVICES",
    "HSA_OVERRIDE_GFX_VERSION",
    "OMP_NUM_THREADS",
    "PYTORCH_ALLOC_CONF",
    "PYTORCH_CUDA_ALLOC_CONF",
    "ROCM_PATH",
    "ROCR_VISIBLE_DEVICES",
)


#: A spawned Python worker writes its own startup messages, and those are not
#: ours to predict: a custom node printing an emoji is ordinary, and on Windows
#: the hidden worker's standard streams are the system code page, so printing
#: one raises UnicodeEncodeError. ComfyUI then tries to log that traceback,
#: hits the same error, and aborts - which reached the API as nothing more
#: informative than "activation failed to start".
#:
#: Set for every Python worker rather than for the packages that happen to
#: print emoji, because the general fact is that a subprocess should be able to
#: write text without the parent's console encoding deciding whether it starts.
_PYTHON_STDIO_KEYS = {
    "PYTHONUTF8": "1",
    "PYTHONIOENCODING": "utf-8",
}


def python_subprocess_environment(
    *,
    overrides: Mapping[str, str] | None = None,
    source: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """A sanitized environment for a Python worker, with text that survives.

    An explicit override wins: these are defaults for workers that say nothing
    about their encoding, never a rewrite of a caller that chose one.
    """
    merged: dict[str, str] = dict(_PYTHON_STDIO_KEYS)
    if overrides:
        merged.update(overrides)
    return subprocess_environment(overrides=merged, source=source)


def subprocess_environment(
    *,
    overrides: Mapping[str, str] | None = None,
    source: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Build a deterministic environment without inherited credentials."""

    parent = os.environ if source is None else source
    environment = {
        key: value
        for key in (*_PASSTHROUGH_KEYS, *_ACCELERATOR_KEYS)
        if (value := parent.get(key)) is not None
    }
    if overrides:
        environment.update(overrides)
    return environment


def git_subprocess_environment() -> dict[str, str]:
    """Build a non-interactive Git environment isolated from user credentials."""

    return subprocess_environment(
        overrides={
            "GIT_ASKPASS": "",
            "GIT_CONFIG_GLOBAL": os.devnull,
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_TERMINAL_PROMPT": "0",
            "SSH_ASKPASS": "",
        }
    )
