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
