"""SAXPY: z = alpha * x + y."""

import numpy as np

from ptxlab.kernels import saxpy
from ptxlab.reference import saxpy as reference_saxpy


def build_kernel():
    return saxpy()


def sample():
    rng = np.random.default_rng(2)
    n = 64
    alpha = np.float32(0.5)
    x = rng.standard_normal(n).astype(np.float32)
    y = rng.standard_normal(n).astype(np.float32)
    z = np.zeros(n, np.float32)
    block = 32
    grid = ((n + block - 1) // block, 1, 1)
    args = {"x": x, "y": y, "z": z, "alpha": float(alpha), "n": n}
    return grid, (block, 1, 1), args, {"z": reference_saxpy(float(alpha), x, y)}
