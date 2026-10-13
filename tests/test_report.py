"""Dashboard view model."""

from pathlib import Path

from ptxlab.report import report_from_path


def test_vector_add_report_matches_and_lists_assumptions() -> None:
    report = report_from_path(Path("examples/vector_add.py"))
    assert report["name"] == "vector_add"
    assert ".version 7.0" in report["ptx"]
    assert "ld_global" in report["ir"]
    assert report["errors"]["c"] == 0.0
    assert report["memory_transactions"] > 0
    assert report["timeline"]
    assert report["trace"]
    assert "not a measured GPU time" in report["assumptions"][0]
    assert report["estimated_cycles"] is not None
