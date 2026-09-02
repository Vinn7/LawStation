"""Pytest-wide isolation for LawStation settings and file audit logging."""

import logging
import os

import pytest


@pytest.fixture(scope="session", autouse=True)
def isolate_audit_log(tmp_path_factory):
    """Keep deliberately failing tests out of the application's real JSONL log."""
    test_log_dir = tmp_path_factory.mktemp("lawstation-test-logs")
    previous_log_dir = os.environ.get("LOG_DIR")
    os.environ["LOG_DIR"] = str(test_log_dir)

    from backend.app.core import logging as app_logging
    from backend.app.core.config import get_settings

    logger = logging.getLogger(app_logging.AUDIT_LOGGER)
    for handler in list(logger.handlers):
        handler.close()
        logger.removeHandler(handler)
    app_logging._configured = False
    get_settings.cache_clear()

    yield

    for handler in list(logger.handlers):
        handler.close()
        logger.removeHandler(handler)
    app_logging._configured = False
    get_settings.cache_clear()
    if previous_log_dir is None:
        os.environ.pop("LOG_DIR", None)
    else:
        os.environ["LOG_DIR"] = previous_log_dir
