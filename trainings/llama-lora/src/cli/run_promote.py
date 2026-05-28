"""Local promotion entrypoint."""

from __future__ import annotations

from common.logging import setup_logging
from training.promote import run


def main() -> None:
    setup_logging("promotion")
    run()


if __name__ == "__main__":
    main()
