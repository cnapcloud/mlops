"""Local training entrypoint."""

from __future__ import annotations

from common.logging import setup_logging
from training.train import run


def main() -> None:
    setup_logging("training")
    run()


if __name__ == "__main__":
    main()
