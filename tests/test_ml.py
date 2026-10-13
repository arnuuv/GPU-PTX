"""Half precision, vector loads, atomics, and row ML kernels."""

import math

import numpy as np

from ptxlab.driver import execute_kernel
from ptxlab.kernels import atomic_count, fma_saxpy, layer_norm, rms_norm, softmax, vector_add_f16, vector_add_v4
from ptxlab.reference import layer_norm as ref_ln
from ptxlab.reference import rms_norm as ref_rms
from ptxlab.reference import softmax_rows


def test_f16_add_rounds_through_f32() -> None:
    a = np.array([1, 2, 0.5], np.float16)
    b = np.array([3, 4, 0.25], np.float16)
    c = np.zeros(3, np.float16)
    execute_kernel(vector_add_f16(), grid=(1, 1, 1), block=(32, 1, 1), args={"a": a, "b": b, "c": c, "n": 3})
    expected = (a.astype(np.float32) + b.astype(np.float32)).astype(np.float16)
    np.testing.assert_array_equal(c, expected)


def test_v4_add_matches_scalar_add() -> None:
    n = 32
    a = np.arange(n, dtype=np.float32)
    b = np.ones(n, np.float32)
    c = np.zeros(n, np.float32)
    execute_kernel(
        vector_add_v4(),
        grid=(1, 1, 1),
        block=(8, 1, 1),
        args={"a": a, "b": b, "c": c, "n": n, "nvec": n // 4},
    )
    np.testing.assert_array_equal(c, a + b)


def test_fma_matches_numpy_fma() -> None:
    x = np.array([1, 2, 3, 4], np.float32)
    y = np.array([0.5, 0.5, 0.5, 0.5], np.float32)
    z = np.zeros(4, np.float32)
    alpha = np.float32(1.25)
    execute_kernel(
        fma_saxpy(),
        grid=(1, 1, 1),
        block=(4, 1, 1),
        args={"x": x, "y": y, "z": z, "alpha": float(alpha), "n": 4},
    )
    expected = np.vectorize(lambda a, b: np.float32(math.fma(float(alpha), float(a), float(b))))(x, y)
    np.testing.assert_array_equal(z, expected)


def test_atomic_count_is_exact_for_integers() -> None:
    n = 40
    counter = np.zeros(1, np.uint32)
    execute_kernel(atomic_count(), grid=(2, 1, 1), block=(32, 1, 1), args={"counter": counter, "n": n})
    assert int(counter[0]) == n


def test_row_kernels_match_their_f32_references() -> None:
    rng = np.random.default_rng(9)
    rows, width = 2, 8
    x = rng.standard_normal((rows, width)).astype(np.float32)
    gamma = rng.standard_normal(width).astype(np.float32)
    beta = rng.standard_normal(width).astype(np.float32)
    weight = rng.standard_normal(width).astype(np.float32)
    y = np.zeros_like(x)
    eps = np.float32(1e-5)
    execute_kernel(
        layer_norm(width),
        grid=(rows, 1, 1),
        block=(width, 1, 1),
        args={"x": x, "gamma": gamma, "beta": beta, "y": y, "rows": rows, "eps": float(eps)},
    )
    np.testing.assert_allclose(y, ref_ln(x, gamma, beta, float(eps)), rtol=0, atol=0)
    y2 = np.zeros_like(x)
    execute_kernel(
        rms_norm(width),
        grid=(rows, 1, 1),
        block=(width, 1, 1),
        args={"x": x, "weight": weight, "y": y2, "rows": rows, "eps": float(eps)},
    )
    np.testing.assert_allclose(y2, ref_rms(x, weight, float(eps)), rtol=0, atol=0)
    y3 = np.zeros_like(x)
    execute_kernel(
        softmax(width),
        grid=(rows, 1, 1),
        block=(width, 1, 1),
        args={"x": x, "y": y3, "rows": rows},
    )
    np.testing.assert_allclose(y3, softmax_rows(x), rtol=1e-5, atol=1e-5)
