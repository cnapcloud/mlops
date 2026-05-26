from data.analysis import _analyze
from data.validation import _run_checks


def test_analyze_reports_duplicates_and_nulls() -> None:
    analysis = _analyze(["hello world", "hello world", None, "  ", "unique text"])

    summary = analysis["summary"]
    assert summary["total_count"] == 5
    assert summary["valid_count"] == 3
    assert summary["null_count"] == 2
    assert summary["duplicate_count"] == 1
    assert summary["unique_duplicate_texts"] == 1


def test_validation_checks_thresholds() -> None:
    checks = _run_checks(
        summary={
            "null_ratio": 0.01,
            "duplicate_ratio": 0.02,
            "valid_count": 100,
            "avg_tokens_estimated": 20,
        },
        max_null_ratio=0.05,
        max_dup_ratio=0.10,
        min_samples=50,
        min_avg_tokens=10,
    )

    assert all(check["passed"] for check in checks.values())
