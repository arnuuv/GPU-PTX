"""Tiled matrix multiplication with a compile-time 8x8 tile."""

import numpy as np

from ptxlab.kernels import tiled_matmul
from ptxlab.reference import matmul as reference_matmul

TILE = 8


def build_kernel():
    return tiled_matmul(TILE)


def sample():
    rng = np.random.default_rng(4)
    rows, depth, cols = 12, 10, 9
    a = rng.standard_normal((rows, depth)).astype(np.float32)
    b = rng.standard_normal((depth, cols)).astype(np.float32)
    c = np.zeros((rows, cols), np.float32)
    grid = ((cols + TILE - 1) // TILE, (rows + TILE - 1) // TILE, 1)
    args = {"a": a, "b": b, "c": c, "m": rows, "n": cols, "k": depth}
    return grid, (TILE, TILE, 1), args, {"c": reference_matmul(a, b)}
