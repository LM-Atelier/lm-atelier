"""Runs before the test package's own conftest, which is the point of it.

Importing the application takes ownership of a data directory, so two test
processes sharing one cannot both start. That is correct for the application -
two instances must not open the same data - and it is why every process in a
parallel run needs a directory of its own. The name has to be settled before
anything imports the application, and the package conftest imports it at module
scope, so this cannot live there.
"""

from __future__ import annotations

import os

#: Set by the distribution plugin in each worker process, absent in a plain run.
WORKER_VARIABLE = "PYTEST_XDIST_WORKER"
DATA_DIRECTORY_VARIABLE = "LOCAL_LM_DATA_DIR"


def _worker_data_directory() -> None:
    worker = os.environ.get(WORKER_VARIABLE)
    if not worker:
        # A single process keeps whatever the caller chose, so an ordinary run
        # and the gate behave exactly as they did.
        return
    configured = os.environ.get(DATA_DIRECTORY_VARIABLE) or "data"
    os.environ[DATA_DIRECTORY_VARIABLE] = f"{configured}-{worker}"


_worker_data_directory()
