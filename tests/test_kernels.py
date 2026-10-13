"""Reduction and tiled matrix multiplication."""

import numpy as np
import pytest

from ptxlab.driver import execute_kernel
from ptxlab.kernels import reduction, tiled_matmul
from ptxlab.reference import block_reduction, matmul


@pytest.mark.parametrize("n,block", [(8, 8), (32, 32), (40, 32), (100, 64)])
def test_reduction_matches_tree_reference(n: int, block: int) -> None:
    rng = np.random.default_rng(n + block)
    src = rng.standard_normal(n).astype(np.float32)
    blocks = (n + block - 1) // block
    dst = np.zeros(blocks, np.float32)
    execute_kernel(
        reduction(block),
        grid=(blocks, 1, 1),
        block=(block, 1, 1),
        args={"src": src, "dst": dst, "n": n, "blocks": blocks},
    )
    np.testing.assert_array_equal(dst, block_reduction(src, block))
    total = np.add.reduce(dst, dtype=np.float32)
    np.testing.assert_allclose(total, np.sum(src, dtype=np.float64), rtol=1e-4, atol=1e-4)


def test_matmul_square_and_ragged() -> None:
    rng = np.random.default_rng(5)
    cases = ((8, 8, 8, 8), (9, 5, 7, 8), (16, 16, 16, 8))
    for rows, depth, cols, tile in cases:
        a = rng.standard_normal((rows, depth)).astype(np.float32)
        b = rng.standard_normal((depth, cols)).astype(np.float32)
        c = np.zeros((rows, cols), np.float32)
        grid = ((cols + tile - 1) // tile, (rows + tile - 1) // tile, 1)
        execute_kernel(
            tiled_matmul(tile),
            grid=grid,
            block=(tile, tile, 1),
            args={"a": a, "b": b, "c": c, "m": rows, "n": cols, "k": depth},
        )
        np.testing.assert_allclose(c, matmul(a, b), rtol=0, atol=0)


def test_identity_matmul() -> None:
    depth = 8
    tile = 8
    eye = np.eye(depth, dtype=np.float32)
    b = np.arange(depth * depth, dtype=np.float32).reshape(depth, depth)
    c = np.zeros_like(b)
    execute_kernel(
        tiled_matmul(tile),
        grid=(1, 1, 1),
        block=(tile, tile, 1),
        args={"a": eye, "b": b, "c": c, "m": depth, "n": depth, "k": depth},
    )
    np.testing.assert_array_equal(c, b)
