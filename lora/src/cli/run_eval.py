"""Local evaluation entrypoint."""

from __future__ import annotations

from common.logging import setup_logging
from training.evaluate import run


def main() -> None:
    setup_logging("evaluation")
    run()


if __name__ == "__main__":
    main()
