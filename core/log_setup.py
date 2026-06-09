"""Logging configuration for SciTracer.

Call ``setup_logging()`` once at startup (in main.py).  After that, every
module can get the shared logger with::

    import logging
    log = logging.getLogger("scitracer")
"""
from __future__ import annotations

import logging
import logging.handlers
from datetime import datetime
from pathlib import Path


def setup_logging(
    log_dir: str | Path = "logs",
    level: int = logging.DEBUG,
    console_level: int = logging.WARNING,
) -> logging.Logger:
    """Configure the ``scitracer`` logger.

    Parameters
    ----------
    log_dir:
        Directory where log files are written.  Created automatically.
    level:
        File handler log level (default DEBUG — captures everything).
    console_level:
        Console handler level (default WARNING — keeps stdout clean).

    Returns
    -------
    The configured ``logging.Logger`` instance.
    """
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file  = log_dir / f"scitracer_{timestamp}.log"

    fmt = logging.Formatter(
        fmt="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    logger = logging.getLogger("scitracer")
    logger.setLevel(logging.DEBUG)

    # File handler — rotating, one new file per run via timestamp in name
    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setLevel(level)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    # Console handler — warnings and above only (doesn't duplicate print output)
    ch = logging.StreamHandler()
    ch.setLevel(console_level)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    logger.info("Log file: %s", log_file)
    return logger
