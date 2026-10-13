"""IR construction, validation, and dumping."""

import pytest

from ptxlab.dtype import DType
from ptxlab.errors import IrError, ToolchainError
from ptxlab.ir import Block, Cbr, Dim, Instr, KernelIR, Op, Param, Ret, VReg
from ptxlab.dump import dump_ir
from ptxlab.toolchain import validate_toolchain
from ptxlab.validate import validate


def _reg(name: str, dtype: DType) -> VReg:
    return VReg(name, dtype)


def test_empty_kernel_validates_and_dumps() -> None:
    kernel = KernelIR(
        name="nop",
        params=[Param("n", DType.U32)],
        shared=[],
        blocks=[Block("entry", [Instr(Op.LD_PARAM, _reg("%n", DType.U32), symbol="n")], Ret())],
    )
    validate(kernel)
    text = dump_ir(kernel)
    assert "kernel nop" in text
    assert "%n = ld_param n" in text
    assert text.strip().endswith("ret")


def test_duplicate_label_and_missing_target() -> None:
    kernel = KernelIR(
        name="bad",
        params=[],
        shared=[],
        blocks=[
            Block("entry", [], Cbr(_reg("%p", DType.PRED), "missing", "missing")),
            Block("entry", [], Ret()),
        ],
    )
    with pytest.raises(IrError, match="duplicate block label"):
        validate(kernel)


def test_type_error_on_mixed_add() -> None:
    kernel = KernelIR(
        name="mix",
        params=[],
        shared=[],
        blocks=[
            Block(
                "entry",
                [
                    Instr(Op.CONST, _reg("%a", DType.F32), args=(1,), dtype=DType.F32),
                    Instr(Op.CONST, _reg("%b", DType.U32), args=(1,), dtype=DType.U32),
                    Instr(
                        Op.ADD,
                        _reg("%c", DType.F32),
                        args=(_reg("%a", DType.F32), _reg("%b", DType.U32)),
                        dtype=DType.F32,
                    ),
                ],
                Ret(),
            )
        ],
    )
    with pytest.raises(IrError, match="add"):
        validate(kernel)


def test_use_before_def() -> None:
    kernel = KernelIR(
        name="undef",
        params=[],
        shared=[],
        blocks=[
            Block(
                "entry",
                [Instr(Op.MOV, _reg("%a", DType.U32), args=(_reg("%b", DType.U32),))],
                Ret(),
            )
        ],
    )
    with pytest.raises(IrError, match="before it is defined"):
        validate(kernel)


def test_toolchain_table() -> None:
    validate_toolchain((7, 0), "sm_70", 64)
    validate_toolchain((7, 8), "sm_90", 64)
    with pytest.raises(ToolchainError, match="address size"):
        validate_toolchain((7, 0), "sm_70", 32)
    with pytest.raises(ToolchainError, match="sm_90"):
        validate_toolchain((7, 0), "sm_90", 64)
    with pytest.raises(ToolchainError, match="outside"):
        validate_toolchain((9, 0), "sm_70", 64)
    with pytest.raises(ToolchainError, match="sm_99"):
        validate_toolchain((7, 0), "sm_99", 64)


def test_pointer_shape_dump() -> None:
    kernel = KernelIR(
        name="vec",
        params=[Param("a", DType.F32, (Dim(name="n"),))],
        shared=[],
        blocks=[
            Block("entry", [Instr(Op.LD_PARAM, _reg("%a", DType.U64), symbol="a")], Ret())
        ],
    )
    validate(kernel)
    assert "param a: ptr f32[n]" in dump_ir(kernel)
