"""Logging helpers used by local scripts and wrappers."""

from __future__ import annotations

import logging
import sys
from datetime import datetime
from common import config

def setup_logging(name: str = "main") -> logging.Logger:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        force=True,
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    return logging.getLogger(name)


def file_logging_path(prefix: str = "pipeline") -> str:
    return f"{prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
