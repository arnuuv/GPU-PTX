"""Elementwise vector addition."""

import numpy as np

from ptxlab.kernels import vector_add
from ptxlab.reference import vector_add as reference_add


def build_kernel():
    return vector_add()


def sample():
    rng = np.random.default_rng(0)
    n = 128
    a = rng.standard_normal(n).astype(np.float32)
    b = rng.standard_normal(n).astype(np.float32)
    c = np.zeros(n, np.float32)
    block = 64
    grid = ((n + block - 1) // block, 1, 1)
    args = {"a": a, "b": b, "c": c, "n": n}
    return grid, block_tuple(block), args, {"c": reference_add(a, b)}


def block_tuple(block: int) -> tuple[int, int, int]:
    return (block, 1, 1)
