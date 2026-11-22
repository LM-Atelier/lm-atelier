"""The part of this process's environment the application reads as its settings."""

from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import contextmanager

from local_lm.runtime_config import RUNTIME_ENV_KEYS


def application_environment() -> dict[str, str]:
    """Every name Settings or the saved runtime settings read, with its value.

    Settings matches its names in any case, so they are found in any case.
    """

    return {
        name: value
        for name, value in os.environ.items()
        if name.upper().startswith("LOCAL_LM_") or name in RUNTIME_ENV_KEYS
    }


@contextmanager
def application_environment_kept() -> Iterator[None]:
    """On leaving, put the application's environment back exactly as it was on entering.

    Whatever wrote to it in between: the launcher, the saved runtime settings,
    or a monkeypatch that cleared a name that was absent and so has nothing to
    undo when the name is written afterwards.
    """

    saved = application_environment()
    try:
        yield
    finally:
        for name in application_environment().keys() - saved.keys():
            del os.environ[name]
        for name, value in saved.items():
            if os.environ.get(name) != value:
                os.environ[name] = value
