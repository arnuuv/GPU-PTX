"""Correctness-first search over launch shapes.

Candidates that fail the reference are recorded as incorrect. When no NVIDIA
device is available, ``ranking`` stays null. Simulator cycle estimates are not
used as a score.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from ptxlab.cuda_host import detect
from ptxlab.driver import compile_kernel, execute_kernel
from ptxlab.kernels import vector_add
from ptxlab.parse import parse
from ptxlab.reference import vector_add as reference_add


@dataclass
class Candidate:
    block: int
    correct: bool
    max_abs_error: float | None
    gpu_median_ms: float | None


def search_vector_add(n: int = 128, blocks: tuple[int, ...] = (32, 64, 128)) -> dict[str, object]:
    status = detect()
    rows = []
    for block in blocks:
        if n % block != 0 and block > n:
            continue
        grid = ((n + block - 1) // block, 1, 1)
        rng = np.random.default_rng(block)
        a = rng.standard_normal(n).astype(np.float32)
        b = rng.standard_normal(n).astype(np.float32)
        c = np.zeros(n, np.float32)
        try:
            execute_kernel(
                vector_add(),
                grid=grid,
                block=(block, 1, 1),
                args={"a": a, "b": b, "c": c, "n": n},
            )
            error = float(np.max(np.abs(c - reference_add(a, b))))
            correct = error == 0.0
        except Exception:
            error = None
            correct = False
        gpu_ms = None
        if correct and status.available:
            gpu_ms = _median_gpu(n, block)
        rows.append(Candidate(block, correct, error, gpu_ms))
    ranking = None
    reason = "No NVIDIA device measurement was collected, so candidates are not ranked by speed."
    measured = [row for row in rows if row.gpu_median_ms is not None]
    if measured:
        ranking = min(measured, key=lambda row: row.gpu_median_ms or 0).block
        reason = "Ranking uses the median of repeated CUDA launch-plus-sync times."
    return {
        "kernel": "vector_add",
        "n": n,
        "dtype": "f32",
        "measurement": "nvidia-gpu" if status.available else "functional-only",
        "gpu": None
        if not status.available
        else {"name": status.device_name, "driver_version": status.driver_version},
        "ranking": ranking,
        "reason": reason,
        "candidates": [asdict(row) for row in rows],
    }


def _median_gpu(n: int, block: int) -> float:
    from ptxlab.cuda_host import launch

    kernel = vector_add()
    program = parse(compile_kernel(kernel))
    ptx = compile_kernel(kernel)
    samples = []
    for _ in range(5):
        a = np.ones(n, np.float32)
        b = np.ones(n, np.float32)
        c = np.zeros(n, np.float32)
        samples.append(
            launch(
                program,
                ptx,
                grid=((n + block - 1) // block, 1, 1),
                block=(block, 1, 1),
                args={"a": a, "b": b, "c": c, "n": n},
            )
        )
    samples.sort()
    return samples[len(samples) // 2]
