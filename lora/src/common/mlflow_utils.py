"""Small MLflow helpers kept out of domain code."""

from __future__ import annotations


def extract_metadata(result: dict, keys: list[str]) -> dict:
    return {key: result.get(key) for key in keys}
