"""Local evaluation entrypoint."""

from __future__ import annotations

from common.logging import setup_logging
import argparse


def main() -> None:
    setup_logging("evaluation")

    parser = argparse.ArgumentParser(description="Run local evaluation")
    parser.add_argument("--run-id", dest="run_id", help="MLflow run id to evaluate (optional)", default=None)
    args = parser.parse_args()

    from training.evaluate import run

    if args.run_id:
        run(train_result={"run_id": args.run_id})
    else:
        run()


if __name__ == "__main__":
    main()
