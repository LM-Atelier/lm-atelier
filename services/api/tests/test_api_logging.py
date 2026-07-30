from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

import pytest

from local_lm.config import Settings
from local_lm.main import API_LOG_BACKUP_COUNT, API_LOG_MAX_BYTES, _api_file_logging


def test_api_file_logging_is_bounded_and_lifecycle_scoped(tmp_path: Path) -> None:
    settings = Settings(data_dir=tmp_path / "data")
    settings.prepare()
    root_logger = logging.getLogger()
    initial_handlers = tuple(root_logger.handlers)

    with (
        pytest.raises(RuntimeError, match="synthetic lifecycle failure"),
        _api_file_logging(settings),
    ):
        handlers = [
            handler
            for handler in root_logger.handlers
            if isinstance(handler, RotatingFileHandler)
            and Path(handler.baseFilename) == settings.log_dir / "api.log"
        ]
        assert len(handlers) == 1
        assert handlers[0].maxBytes == API_LOG_MAX_BYTES
        assert handlers[0].backupCount == API_LOG_BACKUP_COUNT
        logging.getLogger("local_lm.synthetic").warning("synthetic API log marker")
        raise RuntimeError("synthetic lifecycle failure")

    assert tuple(root_logger.handlers) == initial_handlers
    log_path = settings.log_dir / "api.log"
    content = log_path.read_text(encoding="utf-8")
    assert "synthetic API log marker" in content
    assert "LM Atelier application lifecycle failed" in content
    assert "synthetic lifecycle failure" in content
    log_path.replace(settings.log_dir / "api-closed.log")
