"""Local evaluation entrypoint."""

from __future__ import annotations

from common.logging import setup_logging

def main() -> None:
    setup_logging("evaluation")
    
    from training.evaluate import run
    run()


if __name__ == "__main__":
    main()
