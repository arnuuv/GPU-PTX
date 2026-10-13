"""Optimisation passes preserve simulator results."""

import numpy as np

from ptxlab.driver import execute_kernel
from ptxlab.dtype import DType
from ptxlab.dsl import KernelBuilder
from ptxlab.emit import emit
from ptxlab.kernels import vector_add
from ptxlab.lower import lower
from ptxlab.opt import OptimizeConfig, optimize
from ptxlab.parse import parse


def _loopsum():
    k = KernelBuilder("loopsum")
    out = k.array("out", DType.U32, ("n",))
    n = k.scalar("n", DType.U32)
    i = k.let("i", k.gid("x"))
    acc = k.let("acc", k.u32(0))
    with k.for_range("t", 0, 4) as t:
        k.assign(acc, acc + t)
    with k.when(i < n):
        k.store(out, (i,), acc)
    return k.build()


def test_vector_add_matches_with_and_without_opts() -> None:
    from ptxlab.sim import simulate

    a = np.arange(64, dtype=np.float32)
    b = np.ones(64, np.float32)
    plain = np.zeros(64, np.float32)
    raw_out = np.zeros(64, np.float32)
    execute_kernel(
        vector_add(),
        grid=(1, 1, 1),
        block=(64, 1, 1),
        args={"a": a, "b": b, "c": plain, "n": 64},
    )
    simulate(
        parse(emit(lower(vector_add()), allocation="naive")),
        grid=(1, 1, 1),
        block=(64, 1, 1),
        args={"a": a, "b": b, "c": raw_out, "n": 64},
    )
    np.testing.assert_array_equal(plain, a + b)
    np.testing.assert_array_equal(raw_out, plain)


def test_cse_merges_repeated_address_scale() -> None:
    kernel = vector_add()
    raw = emit(lower(kernel), allocation="naive")
    optimised, notes = optimize(lower(kernel))
    text = emit(optimised, allocation="naive")
    assert raw.count(", 4;") > text.count(", 4;")
    assert any(note.startswith("rewrite:") for note in notes)
    assert parse(text).name == "vector_add"


def test_strength_reduction_turns_mul_by_two_into_add() -> None:
    k = KernelBuilder("twice")
    src = k.array("src", DType.U32, ("n",))
    dst = k.array("dst", DType.U32, ("n",))
    n = k.scalar("n", DType.U32)
    i = k.let("i", k.gid("x"))
    with k.when(i < n):
        k.store(dst, (i,), k.load(src, (i,)) * k.u32(2))
    ir, _notes = optimize(lower(k.build()), OptimizeConfig(schedule=False, unroll=False))
    text = emit(ir, allocation="naive")
    assert "mul.lo.u32" not in text.split("st.global")[0] or "add.u32" in text
    src_a = np.arange(16, dtype=np.uint32)
    dst_a = np.zeros(16, np.uint32)
    execute_kernel(k.build(), grid=(1, 1, 1), block=(16, 1, 1), args={"src": src_a, "dst": dst_a, "n": 16})
    np.testing.assert_array_equal(dst_a, (src_a * np.uint32(2)) & np.uint32(0xFFFFFFFF))


def test_unroll_preserves_the_induction_sum() -> None:
    kernel = _loopsum()
    raw = lower(kernel)
    unrolled, notes = optimize(raw, OptimizeConfig(unroll=True, unroll_limit=4))
    assert any(note == "unroll: 1" for note in notes)
    out = np.zeros(8, np.uint32)
    from ptxlab.sim import simulate

    simulate(
        parse(emit(unrolled)),
        grid=(1, 1, 1),
        block=(8, 1, 1),
        args={"out": out, "n": 8},
    )
    np.testing.assert_array_equal(out, np.full(8, 6, np.uint32))


def test_schedule_flag_keeps_results() -> None:
    kernel = vector_add()
    scheduled, notes = optimize(lower(kernel), OptimizeConfig(schedule=True))
    assert any(note.startswith("schedule:") for note in notes)
    a = np.arange(32, dtype=np.float32)
    b = np.arange(32, dtype=np.float32) + np.float32(3)
    c = np.zeros(32, np.float32)
    from ptxlab.sim import simulate

    simulate(parse(emit(scheduled)), grid=(1, 1, 1), block=(32, 1, 1), args={"a": a, "b": b, "c": c, "n": 32})
    np.testing.assert_array_equal(c, a + b)


def test_greedy_allocation_uses_fewer_registers() -> None:
    kernel = vector_add()
    ir, _notes = optimize(lower(kernel))
    naive = parse(emit(ir, allocation="naive"))
    greedy = parse(emit(ir, allocation="greedy"))
    naive_regs = sum(naive.reg_counts.values())
    greedy_regs = sum(greedy.reg_counts.values())
    assert greedy_regs < naive_regs
    a = np.arange(16, dtype=np.float32)
    b = np.ones(16, np.float32)
    left = np.zeros(16, np.float32)
    right = np.zeros(16, np.float32)
    from ptxlab.sim import simulate

    args_l = {"a": a, "b": b, "c": left, "n": 16}
    args_r = {"a": a.copy(), "b": b.copy(), "c": right, "n": 16}
    simulate(parse(emit(ir, allocation="naive")), grid=(1, 1, 1), block=(16, 1, 1), args=args_l)
    simulate(parse(emit(ir, allocation="greedy")), grid=(1, 1, 1), block=(16, 1, 1), args=args_r)
    np.testing.assert_array_equal(left, right)
