"""Functional simulation of the emitted PTX subset."""

import numpy as np
import pytest

from ptxlab.driver import execute_kernel
from ptxlab.dtype import DType
from ptxlab.dsl import KernelBuilder
from ptxlab.errors import DivergentBarrierError, LaunchError, MemoryFault


def _add_kernel():
    k = KernelBuilder("vector_add")
    a = k.array("a", DType.F32, ("n",))
    b = k.array("b", DType.F32, ("n",))
    c = k.array("c", DType.F32, ("n",))
    n = k.scalar("n", DType.U32)
    i = k.let("i", k.gid("x"))
    with k.when(i < n):
        k.store(c, (i,), k.load(a, (i,)) + k.load(b, (i,)))
    return k.build()


def _mul_kernel():
    k = KernelBuilder("vector_mul")
    a = k.array("a", DType.F32, ("n",))
    b = k.array("b", DType.F32, ("n",))
    c = k.array("c", DType.F32, ("n",))
    n = k.scalar("n", DType.U32)
    i = k.let("i", k.gid("x"))
    with k.when(i < n):
        k.store(c, (i,), k.load(a, (i,)) * k.load(b, (i,)))
    return k.build()


def _saxpy_kernel():
    k = KernelBuilder("saxpy")
    x = k.array("x", DType.F32, ("n",))
    y = k.array("y", DType.F32, ("n",))
    z = k.array("z", DType.F32, ("n",))
    alpha = k.scalar("alpha", DType.F32)
    n = k.scalar("n", DType.U32)
    i = k.let("i", k.gid("x"))
    with k.when(i < n):
        k.store(z, (i,), alpha * k.load(x, (i,)) + k.load(y, (i,)))
    return k.build()


def _launch(n: int, block: int) -> tuple[int, int, int]:
    return ((n + block - 1) // block, 1, 1)


@pytest.mark.parametrize("n,block", [(1, 32), (31, 32), (32, 64), (33, 32), (100, 32), (128, 128)])
def test_vector_add_matches_numpy(n: int, block: int) -> None:
    rng = np.random.default_rng(n * 17 + block)
    a = rng.standard_normal(n).astype(np.float32)
    b = rng.standard_normal(n).astype(np.float32)
    c = np.zeros(n, np.float32)
    execute_kernel(_add_kernel(), grid=_launch(n, block), block=(block, 1, 1), args={"a": a, "b": b, "c": c, "n": n})
    np.testing.assert_array_equal(c, np.add(a, b, dtype=np.float32))


@pytest.mark.parametrize("n,block", [(8, 8), (40, 32), (64, 16)])
def test_vector_mul_and_saxpy(n: int, block: int) -> None:
    rng = np.random.default_rng(3)
    a = rng.standard_normal(n).astype(np.float32)
    b = rng.standard_normal(n).astype(np.float32)
    c = np.zeros(n, np.float32)
    execute_kernel(_mul_kernel(), grid=_launch(n, block), block=(block, 1, 1), args={"a": a, "b": b, "c": c, "n": n})
    np.testing.assert_array_equal(c, np.multiply(a, b, dtype=np.float32))
    alpha = np.float32(1.5)
    z = np.zeros(n, np.float32)
    execute_kernel(
        _saxpy_kernel(),
        grid=_launch(n, block),
        block=(block, 1, 1),
        args={"x": a, "y": b, "z": z, "alpha": float(alpha), "n": n},
    )
    np.testing.assert_array_equal(z, np.add(np.multiply(alpha, a, dtype=np.float32), b, dtype=np.float32))


def test_s32_add_is_exact() -> None:
    k = KernelBuilder("sadd")
    a = k.array("a", DType.S32, ("n",))
    b = k.array("b", DType.S32, ("n",))
    c = k.array("c", DType.S32, ("n",))
    n = k.scalar("n", DType.U32)
    i = k.let("i", k.gid("x"))
    with k.when(i < n):
        k.store(c, (i,), k.load(a, (i,)) + k.load(b, (i,)))
    aa = np.array([1, -2, 2**31 - 1], np.int32)
    bb = np.array([3, 5, 1], np.int32)
    cc = np.zeros(3, np.int32)
    execute_kernel(k.build(), grid=(1, 1, 1), block=(32, 1, 1), args={"a": aa, "b": bb, "c": cc, "n": 3})
    np.testing.assert_array_equal(cc, np.add(aa, bb, dtype=np.int32))


def test_divergent_if_writes_both_sides() -> None:
    k = KernelBuilder("sides")
    out = k.array("out", DType.F32, ("n",))
    n = k.scalar("n", DType.U32)
    i = k.let("i", k.gid("x"))
    with k.when(i < n):
        with k.when(i < k.u32(4)) as choice:
            k.store(out, (i,), k.f32(1))
        with choice.otherwise():
            k.store(out, (i,), k.f32(2))
    out = np.zeros(32, np.float32)
    result = execute_kernel(k.build(), grid=(1, 1, 1), block=(32, 1, 1), args={"out": out, "n": 32}, trace=True)
    expected = np.array([1] * 4 + [2] * 28, np.float32)
    np.testing.assert_array_equal(out, expected)
    assert result.divergent_branches >= 1
    assert result.trace


def test_uniform_loop_sums_induction_variable() -> None:
    k = KernelBuilder("loopsum")
    out = k.array("out", DType.U32, ("n",))
    n = k.scalar("n", DType.U32)
    i = k.let("i", k.gid("x"))
    acc = k.let("acc", k.u32(0))
    with k.for_range("t", 0, 4) as t:
        k.assign(acc, acc + t)
    with k.when(i < n):
        k.store(out, (i,), acc)
    out = np.zeros(16, np.uint32)
    result = execute_kernel(k.build(), grid=(1, 1, 1), block=(16, 1, 1), args={"out": out, "n": 16})
    np.testing.assert_array_equal(out, np.full(16, 6, np.uint32))
    assert result.divergent_branches == 0


def test_masked_threads_do_not_fault() -> None:
    k = _add_kernel()
    a = np.arange(10, dtype=np.float32)
    b = np.arange(10, dtype=np.float32)
    c = np.zeros(10, np.float32)
    execute_kernel(k, grid=(1, 1, 1), block=(32, 1, 1), args={"a": a, "b": b, "c": c, "n": 10})
    np.testing.assert_array_equal(c, a + b)


def test_unmasked_oob_load_faults() -> None:
    k = KernelBuilder("oob")
    a = k.array("a", DType.F32, ("n",))
    c = k.array("c", DType.F32, ("n",))
    i = k.let("i", k.gid("x"))
    k.store(c, (i,), k.load(a, (i,)))
    a = np.ones(4, np.float32)
    c = np.zeros(4, np.float32)
    with pytest.raises(MemoryFault):
        execute_kernel(k.build(), grid=(1, 1, 1), block=(8, 1, 1), args={"a": a, "c": c})


def test_launch_rejects_empty_and_oversized_blocks() -> None:
    kernel = _add_kernel()
    args = {
        "a": np.zeros(1, np.float32),
        "b": np.zeros(1, np.float32),
        "c": np.zeros(1, np.float32),
        "n": 1,
    }
    with pytest.raises(LaunchError):
        execute_kernel(kernel, grid=(1, 1, 1), block=(0, 1, 1), args=args)
    with pytest.raises(LaunchError):
        execute_kernel(kernel, grid=(1, 1, 1), block=(32, 33, 1), args=args)


def test_shared_value_is_visible_after_barrier_across_warps() -> None:
    k = KernelBuilder("bcast", expect_block=(64, 1, 1))
    out = k.array("out", DType.F32, ("n",))
    n = k.scalar("n", DType.U32)
    sm = k.shared("sm", DType.F32, (1,))
    i = k.let("i", k.gid("x"))
    tx = k.tid("x")
    with k.when(tx == k.u32(0)):
        k.store_shared(sm, (0,), k.f32(7))
    k.barrier()
    with k.when(i < n):
        k.store(out, (i,), k.load_shared(sm, (0,)))
    out = np.zeros(64, np.float32)
    execute_kernel(k.build(), grid=(1, 1, 1), block=(64, 1, 1), args={"out": out, "n": 64})
    np.testing.assert_array_equal(out, np.full(64, 7, np.float32))


def test_divergent_barrier_is_rejected() -> None:
    k = KernelBuilder("badbar", expect_block=(32, 1, 1))
    tx = k.tid("x")
    with k.when(tx < k.u32(16)):
        k.barrier()
    with pytest.raises(DivergentBarrierError):
        execute_kernel(k.build(), grid=(1, 1, 1), block=(32, 1, 1), args={})
