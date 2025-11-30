"""Logging configuration for the shannon_bench package."""

import coloredlogs


def setup_logging(level: str = "INFO") -> None:
  """Configure the root logger with a short, colored, and tidy format.

  Should be called once at the entry point of the application.

  Args:
    level: Logging level (e.g., "INFO", "DEBUG", "WARNING").
  """
  log_format = (
    "%(asctime)s | <level>%(levelname)-8s</level> | "
    "%(module)s | <level>%(message)s</level>"
  )
  coloredlogs.install(
    level=level,
    fmt=log_format,
    datefmt="%H:%M:%S",
    is_system_wide=True,
  )
