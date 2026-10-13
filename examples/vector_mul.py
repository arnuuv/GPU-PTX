"""Elementwise vector multiplication."""

import numpy as np

from ptxlab.kernels import vector_mul
from ptxlab.reference import vector_mul as reference_mul


def build_kernel():
    return vector_mul()


def sample():
    rng = np.random.default_rng(1)
    n = 96
    a = rng.standard_normal(n).astype(np.float32)
    b = rng.standard_normal(n).astype(np.float32)
    c = np.zeros(n, np.float32)
    block = 32
    grid = ((n + block - 1) // block, 1, 1)
    return grid, (block, 1, 1), {"a": a, "b": b, "c": c, "n": n}, {"c": reference_mul(a, b)}
