import logging
import os
from pathlib import Path

from backend.app.core.logging import AUDIT_LOGGER, audit


def test_pytest_audit_log_uses_temporary_directory():
    audit("test.audit.isolated", status="success")

    expected_dir = Path(os.environ["LOG_DIR"]).resolve()
    file_handlers = [
        handler
        for handler in logging.getLogger(AUDIT_LOGGER).handlers
        if hasattr(handler, "baseFilename")
    ]

    assert file_handlers
    assert all(
        Path(handler.baseFilename).resolve().parent == expected_dir
        for handler in file_handlers
    )
    assert expected_dir != (Path.cwd() / "data/logs").resolve()
