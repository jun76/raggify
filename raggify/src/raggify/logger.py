from __future__ import annotations

import logging
import traceback
from typing import Literal

from rich.console import Console

from .core.const import PROJECT_NAME

__all__ = ["configure_logging", "logger", "console"]


class Color:
    ResetAll = "\033[0m"

    Bold = "\033[1m"
    Dim = "\033[2m"
    Underlined = "\033[4m"
    Blink = "\033[5m"
    Reverse = "\033[7m"
    Hidden = "\033[8m"

    ResetBold = "\033[21m"
    ResetDim = "\033[22m"
    ResetUnderlined = "\033[24m"
    ResetBlink = "\033[25m"
    ResetReverse = "\033[27m"
    ResetHidden = "\033[28m"

    Default = "\033[39m"
    Black = "\033[30m"
    Red = "\033[31m"
    Green = "\033[32m"
    Yellow = "\033[33m"
    Blue = "\033[34m"
    Magenta = "\033[35m"
    Cyan = "\033[36m"
    LightGray = "\033[37m"
    DarkGray = "\033[90m"
    LightRed = "\033[91m"
    LightGreen = "\033[92m"
    LightYellow = "\033[93m"
    LightBlue = "\033[94m"
    LightMagenta = "\033[95m"
    LightCyan = "\033[96m"
    White = "\033[97m"


_LOG_FORMAT = (
    f"{Color.Blue}%(levelname)s{Color.ResetAll}: "
    f"{Color.DarkGray}%(asctime)s "
    f"{Color.DarkGray}%(name)s "
    f"{Color.White}%(message)s "
    f"{Color.DarkGray}@ %(pathname)s:%(lineno)d %(funcName)s "
    f"{Color.ResetAll}"
)


class ShortTracebackFormatter(logging.Formatter):
    """Custom formatter to shorten tracebacks in log messages."""

    def formatException(self, ei):
        exc_type, exc_value, exc_tb = ei

        if exc_type is None or exc_value is None:
            return super().formatException(ei)

        te = traceback.TracebackException(
            exc_type,
            exc_value,
            exc_tb,
            limit=2,
            capture_locals=False,
        )
        return "".join(te.format())


logger = logging.getLogger(PROJECT_NAME)


def configure_logging(
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO",
) -> None:
    """Configure logging so library logs stay at INFO while raggify honors log_level."""
    logger.handlers.clear()
    logger.setLevel(log_level)
    logger.propagate = False

    app_handler = logging.StreamHandler()
    app_handler.setLevel(log_level)
    app_handler.setFormatter(ShortTracebackFormatter(_LOG_FORMAT))
    logger.addHandler(app_handler)


console = Console()
