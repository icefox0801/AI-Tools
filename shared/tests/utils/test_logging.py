"""
Unit tests for shared.utils.logging module.
"""

import logging
import os
from unittest.mock import patch

import pytest

from shared.utils.logging import DEFAULT_FORMAT, get_logger, set_log_level, setup_logging


class TestSetupLogging:
    """Tests for setup_logging function."""

    def test_setup_logging_returns_logger(self):
        """Test setup_logging returns a logger instance."""
        logger = setup_logging("test_logger")
        assert isinstance(logger, logging.Logger)
        assert logger.name == "test_logger"

    def test_setup_logging_with_custom_level(self):
        """Test setup_logging with custom level."""
        logger = setup_logging("test_custom_level", level="DEBUG")
        assert logger.level == logging.DEBUG

    def test_setup_logging_default_level_info(self):
        """Test default level is INFO when no env var set."""
        with patch.dict(os.environ, {}, clear=True):
            # Remove LOG_LEVEL if present
            os.environ.pop("LOG_LEVEL", None)
            logger = setup_logging("test_default_level")
            assert logger.level == logging.INFO

    def test_setup_logging_from_env_var(self):
        """Test setup_logging reads LOG_LEVEL from env."""
        with patch.dict(os.environ, {"LOG_LEVEL": "WARNING"}):
            logger = setup_logging("test_env_level")
            assert logger.level == logging.WARNING

    def test_setup_logging_none_name_returns_root(self):
        """Test None name returns root logger."""
        logger = setup_logging(None)
        assert logger.name == "root"


class TestGetLogger:
    """Tests for get_logger function."""

    def test_get_logger_returns_named_logger(self):
        """Test get_logger returns logger with specified name."""
        logger = get_logger("my.module")
        assert isinstance(logger, logging.Logger)
        assert logger.name == "my.module"

    def test_get_logger_returns_same_instance(self):
        """Test get_logger returns same logger for same name."""
        logger1 = get_logger("same_name")
        logger2 = get_logger("same_name")
        assert logger1 is logger2


class TestSetLogLevel:
    """Tests for set_log_level function."""

    def test_set_log_level_debug(self):
        """Test setting log level to DEBUG."""
        set_log_level("DEBUG")
        assert logging.getLogger().level == logging.DEBUG

    def test_set_log_level_warning(self):
        """Test setting log level to WARNING."""
        set_log_level("WARNING")
        assert logging.getLogger().level == logging.WARNING

    def test_set_log_level_case_insensitive(self):
        """Test log level is case insensitive."""
        set_log_level("error")
        assert logging.getLogger().level == logging.ERROR


class TestDefaultFormat:
    """Tests for DEFAULT_FORMAT constant."""

    def test_default_format_contains_required_fields(self):
        """Test DEFAULT_FORMAT has all required fields."""
        assert "%(asctime)s" in DEFAULT_FORMAT
        assert "%(name)s" in DEFAULT_FORMAT
        assert "%(levelname)s" in DEFAULT_FORMAT
        assert "%(message)s" in DEFAULT_FORMAT


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
