"""AST builder and IR lowering."""

import pytest

from ptxlab.dtype import DType
from ptxlab.dsl import KernelBuilder
from ptxlab.dump import dump_ir
from ptxlab.errors import LoweringError, ToolchainError
from ptxlab.lower import lower


def vector_add():
    k = KernelBuilder("vector_add")
    a = k.array("a", DType.F32, ("n",))
    b = k.array("b", DType.F32, ("n",))
    c = k.array("c", DType.F32, ("n",))
    n = k.scalar("n", DType.U32)
    i = k.let("i", k.gid("x"))
    with k.when(i < n):
        k.store(c, (i,), k.load(a, (i,)) + k.load(b, (i,)))
    return k.build()


def test_vector_add_ir_has_bounds_check_and_addressing() -> None:
    ir = lower(vector_add())
    text = dump_ir(ir)
    assert "cmp.lt.u32 %i, %n" in text or "cmp.ge" in text or "cmp.lt.u32" in text
    assert "ld_global" in text
    assert "st_global" in text
    assert "mul_wide.u32" in text
    assert "cbr" in text
    assert text.strip().endswith("ret")


def test_mixed_arithmetic_is_rejected() -> None:
    k = KernelBuilder("mixed")
    out = k.array("out", DType.F32, ("n",))
    n = k.scalar("n", DType.U32)
    i = k.let("i", k.gid("x"))
    with pytest.raises(LoweringError, match="expected f32"):
        k.store(out, (i,), k.load(out, (i,)) + n)


def test_for_loop_lowers_to_a_header_and_back_edge() -> None:
    k = KernelBuilder("loopsum")
    out = k.array("out", DType.U32, ("n",))
    n = k.scalar("n", DType.U32)
    i = k.let("i", k.gid("x"))
    acc = k.let("acc", k.u32(0))
    with k.for_range("t", 0, 4) as t:
        k.assign(acc, acc + t)
    with k.when(i < n):
        k.store(out, (i,), acc)
    ir = lower(k.build())
    text = dump_ir(ir)
    assert "L_for_0:" in text
    assert "br L_for_0" in text
    assert "add.u32 %acc, %t" in text or "add.u32 %acc" in text


def test_rejected_target_fails_during_lowering() -> None:
    k = KernelBuilder("nope", version=(7, 0), target="sm_90")
    k.scalar("n", DType.U32)
    with pytest.raises(ToolchainError, match="sm_90"):
        lower(k.build())


def test_predicate_expression_is_not_a_python_bool() -> None:
    k = KernelBuilder("pred")
    n = k.scalar("n", DType.U32)
    with pytest.raises(TypeError):
        if n < k.u32(1):
            pass
