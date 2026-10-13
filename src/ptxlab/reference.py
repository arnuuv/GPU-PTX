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


def tree_reduce(values: np.ndarray, combine) -> np.float32:
    acc = np.array(values, dtype=np.float32, copy=True)
    offset = acc.size // 2
    while offset:
        acc[:offset] = combine(acc[:offset], acc[offset : 2 * offset])
        offset //= 2
    return np.float32(acc[0])


def _fadd(left, right):
    return np.add(left, right, dtype=np.float32)


def _fmax(left, right):
    return np.maximum(left, right).astype(np.float32)


def _row_norm_stats(row: np.ndarray):
    total = tree_reduce(row, _fadd)
    mean = np.divide(total, np.float32(row.size), dtype=np.float32)
    centered = np.subtract(row, mean, dtype=np.float32)
    var = np.divide(tree_reduce(np.multiply(centered, centered, dtype=np.float32), _fadd), np.float32(row.size), dtype=np.float32)
    return mean, var, centered


def layer_norm(x: np.ndarray, gamma: np.ndarray, beta: np.ndarray, eps: float) -> np.ndarray:
    out = np.empty_like(x)
    eps_f = np.float32(eps)
    for index, row in enumerate(x):
        mean, var, centered = _row_norm_stats(row)
        inv = np.divide(np.float32(1), np.sqrt(np.add(var, eps_f, dtype=np.float32)), dtype=np.float32)
        scaled = np.multiply(np.multiply(centered, inv, dtype=np.float32), gamma, dtype=np.float32)
        out[index] = np.add(scaled, beta, dtype=np.float32)
    return out


def rms_norm(x: np.ndarray, weight: np.ndarray, eps: float) -> np.ndarray:
    out = np.empty_like(x)
    eps_f = np.float32(eps)
    for index, row in enumerate(x):
        mean_sq = np.divide(tree_reduce(np.multiply(row, row, dtype=np.float32), _fadd), np.float32(row.size), dtype=np.float32)
        inv = np.divide(np.float32(1), np.sqrt(np.add(mean_sq, eps_f, dtype=np.float32)), dtype=np.float32)
        out[index] = np.multiply(np.multiply(row, inv, dtype=np.float32), weight, dtype=np.float32)
    return out


def softmax_rows(x: np.ndarray) -> np.ndarray:
    """Matches the interpreter's flushed exp2, not a GPU special-function unit."""

    out = np.empty_like(x)
    log2e = np.float32(np.log2(np.e))
    tiny = np.finfo(np.float32).tiny
    for index, row in enumerate(x):
        peak = tree_reduce(row, _fmax)
        shifted = np.multiply(np.subtract(row, peak, dtype=np.float32), log2e, dtype=np.float32)
        expo = np.exp2(shifted).astype(np.float32)
        expo = np.where((expo != 0) & (np.abs(expo) < tiny), np.float32(0), expo)
        total = tree_reduce(expo, _fadd)
        out[index] = np.divide(expo, total, dtype=np.float32)
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
