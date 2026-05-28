"""Local data analysis entrypoint."""

from __future__ import annotations

from common.logging import setup_logging
from data.analysis import run


def main() -> None:
    setup_logging("data.analysis")
    run()


if __name__ == "__main__":
    main()
