"""
JSON-structured logger using loguru.
All trading events are logged in machine-parseable JSON.
"""

import sys
from pathlib import Path

from loguru import logger

from quanti.config import settings


def setup_logger(
    name: str = "quanti",
    log_dir: str | None = None,
) -> None:
    """
    Configure structured JSON logging.

    Outputs:
    - stdout: human-readable (colorized)
    - file: JSON-structured with rotation
    """
    log_dir = Path(log_dir or settings.LOG_DIR)
    log_dir.mkdir(parents=True, exist_ok=True)

    # Remove default handler
    logger.remove()

    # Console: human-readable
    logger.add(
        sys.stdout,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan> | <level>{message}</level>",
        level="INFO",
        colorize=True,
    )

    # File: JSON structured
    logger.add(
        log_dir / f"{name}.jsonl",
        format="{message}",
        level="INFO",
        rotation=f"{settings.LOG_MAX_SIZE_MB} MB",
        retention=settings.LOG_BACKUP_COUNT,
        serialize=True,  # JSON output
    )

    # Error file: only warnings and above
    logger.add(
        log_dir / f"{name}_errors.jsonl",
        format="{message}",
        level="WARNING",
        rotation=f"{settings.LOG_MAX_SIZE_MB} MB",
        retention=settings.LOG_BACKUP_COUNT,
        serialize=True,
    )

    logger.info("Logger initialized", module="monitor.logger")


def get_logger(name: str = "quanti"):
    """Get a bound logger instance for a specific module."""
    return logger.bind(name=name)
