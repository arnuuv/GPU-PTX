"""Autotune and the benchmark report do not invent GPU timings."""

import json

from ptxlab.autotune import search_vector_add
from ptxlab.benchreport import build_report
from ptxlab.cuda_host import detect


def test_search_checks_correctness_and_does_not_rank_without_a_gpu() -> None:
    report = search_vector_add(n=64, blocks=(32, 64))
    assert report["candidates"]
    assert all(row["correct"] and row["max_abs_error"] == 0.0 for row in report["candidates"])
    status = detect()
    if status.available:
        assert report["measurement"] == "nvidia-gpu"
        assert report["ranking"] in {32, 64}
    else:
        assert report["measurement"] == "functional-only"
        assert report["ranking"] is None
        assert all(row["gpu_median_ms"] is None for row in report["candidates"])


def test_report_separates_interpreter_model_and_gpu(tmp_path) -> None:
    report = build_report(64, 32)
    assert report["interpreter"]["max_abs_error"] == 0.0
    assert report["interpreter"]["source"] == "cpu-interpreter"
    assert report["analytical"]["source"] == "analytical-estimate"
    assert "not a measured GPU time" in report["analytical"]["assumptions"][0]
    status = detect()
    if not status.available:
        assert report["gpu"]["measured"] is False
        assert report["gpu"]["latency_ms"] is None
    path = tmp_path / "report.json"
    path.write_text(json.dumps(report), encoding="utf-8")
    assert "analytical-estimate" in path.read_text(encoding="utf-8")
