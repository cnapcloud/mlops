"""CLI entrypoint for seeding initial MinIO data."""

from __future__ import annotations

from common.logging import setup_logging
from data.seed import run


def main() -> None:
    setup_logging("seed")
    run()


if __name__ == "__main__":
    main()