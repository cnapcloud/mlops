"""Logging helpers used by local scripts and wrappers."""

from __future__ import annotations

import logging
import os
import sys
from datetime import datetime
from common import config

def setup_logging(name: str = "main") -> logging.Logger:
    is_airflow = "AIRFLOW" in os.environ

    if is_airflow:
        # Airflow가 사용하는 루트 로거의 핸들러를 그대로 가져옵니다.
        # 이렇게 하면 Airflow의 로깅 설정(포맷, 출력처)을 그대로 상속받습니다.
        logger = logging.getLogger(name)
        logger.setLevel(logging.INFO)
        # 중요: 상위 Airflow 로거로 로그를 전파시킵니다.
        logger.propagate = True 
        return logger
    else:
        # 로컬 실행 시 기존 기본 설정 사용
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
