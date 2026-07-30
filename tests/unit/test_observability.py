from __future__ import annotations

import io
import logging

import pytest

from steering.observability import PACKAGE_LOGGER, configure_logging


@pytest.fixture(autouse=True)
def restore_package_logger() -> object:
    logger = logging.getLogger(PACKAGE_LOGGER)
    handlers, level, propagate = list(logger.handlers), logger.level, logger.propagate
    yield
    logger.handlers = handlers
    logger.setLevel(level)
    logger.propagate = propagate


def test_configured_logging_emits_package_records() -> None:
    stream = io.StringIO()
    configure_logging("info", stream=stream)

    logging.getLogger("steering.ingestion.x").info("resolved post %s", "123")

    assert "resolved post 123" in stream.getvalue()
    assert "steering.ingestion.x" in stream.getvalue()


def test_level_filtering_is_applied() -> None:
    stream = io.StringIO()
    configure_logging("warning", stream=stream)
    logger = logging.getLogger("steering.ingestion.browser")

    logger.info("quiet detail")
    logger.warning("loud problem")

    assert "quiet detail" not in stream.getvalue()
    assert "loud problem" in stream.getvalue()


def test_repeated_configuration_does_not_duplicate_output() -> None:
    stream = io.StringIO()
    configure_logging("info", stream=stream)
    configure_logging("info", stream=stream)

    logging.getLogger("steering.test").info("once")

    assert stream.getvalue().count("once") == 1


def test_the_root_logger_is_left_to_the_host_application() -> None:
    stream = io.StringIO()
    configure_logging("debug", stream=stream)

    logging.getLogger("some.other.library").warning("not ours")

    assert stream.getvalue() == ""


def test_an_unknown_level_is_rejected() -> None:
    with pytest.raises(ValueError, match="log level must be one of"):
        configure_logging("chatty")
