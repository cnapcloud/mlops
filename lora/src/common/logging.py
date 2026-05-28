"""Logging helpers used by local scripts and wrappers."""

from __future__ import annotations

import logging
import os
import sys
from datetime import datetime
from common import config

def setup_logging(name: str = "main") -> logging.Logger:
    # PodOperator에서 주입한 환경변수 확인
    is_airflow = os.environ.get("AIRFLOW") == "True"

    if is_airflow:
        log_format = "[%(levelname)s] %(name)s - %(message)s"
    else:
        log_format = "%(asctime)s [%(levelname)s] %(name)s - %(message)s"

    # 무조건 StreamHandler(sys.stdout)를 통해 화면에 로그를 뱉어내야
    # 쿠버네티스가 이를 가로채서 Airflow UI로 전달할 수 있습니다.
    logging.basicConfig(
        level=logging.INFO,
        format=log_format,
        datefmt="%Y-%m-%d %H:%M:%S",
        force=True,
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    return logging.getLogger(name)


def file_logging_path(prefix: str = "pipeline") -> str:
    return f"{prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
