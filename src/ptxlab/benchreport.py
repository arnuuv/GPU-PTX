"""Write a report that keeps interpreter, model, and GPU results apart."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ptxlab.cuda_host import detect
from ptxlab.driver import compile_kernel, execute_kernel
from ptxlab.kernels import vector_add
from ptxlab.parse import parse
from ptxlab.perf import estimate
from ptxlab.reference import vector_add as reference_add


def build_report(n: int = 128, block: int = 64) -> dict[str, object]:
    rng = np.random.default_rng(0)
    a = rng.standard_normal(n).astype(np.float32)
    b = rng.standard_normal(n).astype(np.float32)
    c = np.zeros(n, np.float32)
    grid = ((n + block - 1) // block, 1, 1)
    kernel = vector_add()
    result = execute_kernel(kernel, grid=grid, block=(block, 1, 1), args={"a": a, "b": b, "c": c, "n": n})
    error = float(np.max(np.abs(c - reference_add(a, b))))
    program = parse(compile_kernel(kernel))
    model = estimate(program, result, grid=(grid[0], 1, 1), block=(block, 1, 1))
    status = detect()
    return {
        "kernel": "vector_add",
        "n": n,
        "dtype": "f32",
        "block": [block, 1, 1],
        "interpreter": {
            "max_abs_error": error,
            "instructions": result.instructions,
            "source": "cpu-interpreter",
        },
        "analytical": {
            "estimated_cycles": model.estimated_cycles,
            "memory_transactions": model.memory_transactions,
            "occupancy": model.occupancy,
            "assumptions": list(model.assumptions),
            "source": "analytical-estimate",
        },
        "gpu": {
            "measured": False,
            "reason": status.reason,
            "latency_ms": None,
            "source": "nvidia-gpu" if status.available else "unavailable",
        },
    }


def write_report(path: Path, n: int = 128) -> dict[str, object]:
    report = build_report(n)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report
