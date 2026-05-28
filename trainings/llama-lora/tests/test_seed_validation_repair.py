from data.analysis import _analyze
from data.seed_data import _build_seed_data
from data.validation_data import _repair_texts, _run_checks


def test_seed_data_uses_expected_shape() -> None:
    seed_data = _build_seed_data()

    assert len(seed_data) == 100
    assert sum(item is None for item in seed_data) == 4


def test_validation_repair_fixes_nulls_duplicates_and_sample_count() -> None:
    source_texts = ["hello world", "hello world", None, "", "short text"]
    source_checks = _run_checks(
        summary=_analyze(source_texts)["summary"],
        max_null_ratio=0.05,
        max_dup_ratio=0.10,
        min_samples=10,
        min_avg_tokens=10,
    )

    repaired_texts, repair_actions = _repair_texts(
        source_texts,
        source_checks,
        min_samples=10,
        min_avg_tokens=10,
    )

    repaired_summary = _analyze(repaired_texts)["summary"]
    repaired_checks = _run_checks(
        summary=repaired_summary,
        max_null_ratio=0.05,
        max_dup_ratio=0.10,
        min_samples=10,
        min_avg_tokens=10,
    )

    assert repaired_checks["null_ratio"]["passed"]
    assert repaired_checks["duplicate_ratio"]["passed"]
    assert repaired_checks["min_samples"]["passed"]
    assert repaired_checks["avg_tokens"]["passed"]
    assert repair_actions
    assert all(isinstance(item, str) and item.strip() for item in repaired_texts)