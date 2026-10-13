"""NumPy references that follow the same f32 association as the kernels."""

from __future__ import annotations

import numpy as np


def vector_add(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.add(a, b, dtype=np.float32)


def vector_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.multiply(a, b, dtype=np.float32)


def saxpy(alpha: float, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    scaled = np.multiply(np.float32(alpha), x, dtype=np.float32)
    return np.add(scaled, y, dtype=np.float32)


def block_reduction(src: np.ndarray, block_size: int) -> np.ndarray:
    """One partial per block, using the same pairwise tree as ``kernels.reduction``."""

    blocks = (src.size + block_size - 1) // block_size
    out = np.zeros(blocks, np.float32)
    for block in range(blocks):
        shared = np.zeros(block_size, np.float32)
        start = block * block_size
        chunk = src[start : start + block_size]
        shared[: chunk.size] = chunk
        offset = block_size // 2
        while offset:
            shared[:offset] = np.add(shared[:offset], shared[offset : 2 * offset], dtype=np.float32)
            offset //= 2
        out[block] = shared[0]
    return out


def matmul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Row-major GEMM accumulated in binary32, K running from 0."""

    rows, depth = a.shape
    cols = b.shape[1]
    out = np.zeros((rows, cols), np.float32)
    for row in range(rows):
        for col in range(cols):
            acc = np.float32(0)
            for k in range(depth):
                acc = np.float32(acc + np.float32(a[row, k] * b[k, col]))
            out[row, col] = acc
    return out
