"""PTX emission and the restricted parser."""

import pytest

from ptxlab.dtype import DType
from ptxlab.dsl import KernelBuilder
from ptxlab.emit import IR_PTX_MNEMONIC, emit
from ptxlab.errors import ParseError, UnsupportedInstruction
from ptxlab.ir import Op
from ptxlab.lower import lower
from ptxlab.parse import parse


def _vector_add():
    k = KernelBuilder("vector_add")
    a = k.array("a", DType.F32, ("n",))
    b = k.array("b", DType.F32, ("n",))
    c = k.array("c", DType.F32, ("n",))
    n = k.scalar("n", DType.U32)
    i = k.let("i", k.gid("x"))
    with k.when(i < n):
        k.store(c, (i,), k.load(a, (i,)) + k.load(b, (i,)))
    return lower(k.build())


def test_every_ir_op_has_a_ptx_mnemonic() -> None:
    assert set(IR_PTX_MNEMONIC) == set(Op)


def test_vector_add_ptx_round_trips() -> None:
    text = emit(_vector_add())
    program = parse(text)
    ops = [inst.op for inst in program.insts]
    assert program.name == "vector_add"
    assert program.version == (7, 0)
    assert program.target == "sm_70"
    assert program.address_size == 64
    assert [param.kind for param in program.params] == ["u64", "u64", "u64", "u32"]
    for required in (
        "mov.u32",
        "mul.lo.u32",
        "setp.lt.u32",
        "mul.wide.u32",
        "add.u64",
        "ld.param.u64",
        "ld.global.f32",
        "add.rn.f32",
        "st.global.f32",
        "bra",
        "ret",
    ):
        assert required in ops
    assert "%tid.x" in text and "%ctaid.x" in text and "%ntid.x" in text
    assert "entry:" in text
    assert program.labels["entry"] == 0


def test_coverage_kernel_parses_optional_ops() -> None:
    k = KernelBuilder("coverage", expect_block=(32, 1, 1))
    a = k.array("a", DType.F32, ("n",))
    b = k.array("b", DType.F32, ("n",))
    c = k.array("c", DType.F32, ("n",))
    x = k.array("x", DType.S32, ("n",))
    y = k.array("y", DType.S32, ("n",))
    z = k.array("z", DType.S32, ("n",))
    flags = k.array("flags", DType.U32, ("n",))
    n = k.scalar("n", DType.U32)
    sm = k.shared("smem", DType.F32, (32,))
    i = k.let("i", k.gid("x"))
    with k.when(i < n):
        k.store(c, (i,), (k.load(a, (i,)) - k.load(b, (i,))) * k.f32(2.0) + k.load(a, (i,)))
        k.store(z, (i,), k.mad_lo(k.load(x, (i,)), k.load(y, (i,)), k.s32(1)))
        k.store(flags, (i,), k.shr(i, 1))
    with k.when(k.land(i < n, k.lor(i == k.u32(0), k.lnot(i < k.u32(2))))):
        k.store_shared(sm, (k.tid("x"),), k.load(a, (i,)))
    k.barrier()
    text = emit(lower(k.build()))
    program = parse(text)
    ops = {inst.op for inst in program.insts}
    for required in (
        "sub.rn.f32",
        "mul.rn.f32",
        "mad.lo.s32",
        "shr.u32",
        "and.pred",
        "or.pred",
        "not.pred",
        "st.shared.f32",
        "bar.sync",
    ):
        assert required in ops
    assert program.expect_block == (32, 1, 1)
    assert program.shared[0].name == "smem"
    assert program.shared[0].length == 32


def test_unsupported_instruction_is_rejected() -> None:
    text = emit(_vector_add()).replace("ld.global.f32", "ld.global.v2.f32", 1)
    with pytest.raises(UnsupportedInstruction, match="ld.global.v2.f32"):
        parse(text)


def test_bad_address_size_is_rejected() -> None:
    text = emit(_vector_add()).replace(".address_size 64", ".address_size 32")
    with pytest.raises(Exception, match="address size"):
        parse(text)


def test_unknown_label_is_rejected() -> None:
    text = emit(_vector_add()).replace("bra L_then_0", "bra L_missing", 1)
    with pytest.raises(ParseError, match="unknown label"):
        parse(text)
