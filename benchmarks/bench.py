"""Reproducible analytical benchmark.

Prints functional error against the NumPy reference and the analytical cycle
estimate. Nothing here is a measured GPU time or a speedup.
"""

from __future__ import annotations

import numpy as np

from ptxlab.driver import compile_kernel, execute_kernel
from ptxlab.kernels import saxpy, vector_add, vector_mul
from ptxlab.parse import parse
from ptxlab.perf import estimate
from ptxlab.reference import saxpy as ref_saxpy
from ptxlab.reference import vector_add as ref_add
from ptxlab.reference import vector_mul as ref_mul


def main() -> None:
    print("PTXLab analytical benchmark")
    print("Cycles are model estimates, not GPU measurements. No speedup is claimed.")
    print(f"{'kernel':<16} {'n':>8} {'max_abs_err':>14} {'transactions':>14} {'est_cycles':>12} {'occupancy':>10}")
    for name, builder, reference, n, block in (
        ("vector_add", vector_add, ref_add, 256, 64),
        ("vector_mul", vector_mul, ref_mul, 256, 64),
        ("saxpy", saxpy, None, 256, 64),
    ):
        rng = np.random.default_rng(n)
        left = rng.standard_normal(n).astype(np.float32)
        right = rng.standard_normal(n).astype(np.float32)
        out = np.zeros(n, np.float32)
        grid = ((n + block - 1) // block, 1, 1)
        kernel = builder()
        if name == "saxpy":
            alpha = np.float32(1.25)
            args = {"x": left, "y": right, "z": out, "alpha": float(alpha), "n": n}
            expected = ref_saxpy(float(alpha), left, right)
        else:
            args = {"a": left, "b": right, "c": out, "n": n}
            expected = reference(left, right)
        result = execute_kernel(kernel, grid=grid, block=(block, 1, 1), args=args)
        program = parse(compile_kernel(kernel))
        stats = estimate(program, result, grid=(grid[0], 1, 1), block=(block, 1, 1))
        error = float(np.max(np.abs(out.astype(np.float64) - expected.astype(np.float64))))
        print(
            f"{name:<16} {n:8d} {error:14.3e} {stats.memory_transactions:14d} "
            f"{stats.estimated_cycles:12d} {stats.occupancy:10.3f}"
        )


if __name__ == "__main__":
    main()
