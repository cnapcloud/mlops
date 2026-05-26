"""Local data validation entrypoint."""

from __future__ import annotations

from common.logging import setup_logging
from data.validation import run


def main() -> None:
    setup_logging("data.validation")
    run()


if __name__ == "__main__":
    main()
