"""Emit the supported NVIDIA PTX subset from typed IR.

Floating-point arithmetic uses round-to-nearest-even (``.rn``), which PTX ISA 7.0
requires on ``add`` and ``mul`` of ``.f32``. Integer multiplies use ``.lo`` or
``mul.wide``. Global addresses are 64-bit.
"""

from __future__ import annotations

import numpy as np

from ptxlab.dtype import DType
from ptxlab.errors import EmitError
from ptxlab.ir import Cbr, Instr, KernelIR, Op, Ret, VReg
from ptxlab.toolchain import format_version
from ptxlab.validate import validate

# IR operation -> PTX mnemonic. Types select the full opcode in ``_opcode``.
IR_PTX_MNEMONIC: dict[Op, str] = {
    Op.CONST: "mov",
    Op.MOV: "mov",
    Op.ADD: "add",
    Op.SUB: "sub",
    Op.MUL: "mul",
    Op.MUL_WIDE: "mul.wide",
    Op.MAD_LO: "mad.lo",
    Op.FMA: "fma",
    Op.DIV: "div",
    Op.SQRT: "sqrt",
    Op.MAX: "max",
    Op.EX2: "ex2",
    Op.CVT: "cvt",
    Op.SHR: "shr",
    Op.CMP: "setp",
    Op.AND_PRED: "and.pred",
    Op.OR_PRED: "or.pred",
    Op.NOT_PRED: "not.pred",
    Op.TID: "mov",
    Op.CTAID: "mov",
    Op.NTID: "mov",
    Op.LD_PARAM: "ld.param",
    Op.LD_GLOBAL: "ld.global",
    Op.ST_GLOBAL: "st.global",
    Op.LD_GLOBAL_V4: "ld.global.v4",
    Op.ST_GLOBAL_V4: "st.global.v4",
    Op.ATOM_ADD: "atom.add",
    Op.SHARED_BASE: "mov",
    Op.LD_SHARED: "ld.shared",
    Op.ST_SHARED: "st.shared",
    Op.BAR: "bar.sync",
}


class _Alloc:
    def __init__(self) -> None:
        self.names: dict[str, str] = {}
        self.counts: dict[DType, int] = {dtype: 0 for dtype in DType}

    def preload(self, mapping: dict[str, str]) -> None:
        self.names.update(mapping)
        for physical in mapping.values():
            for dtype in DType:
                prefix = dtype.ptx_reg_prefix
                if physical.startswith(prefix) and physical.removeprefix(prefix).isdigit():
                    index = int(physical.removeprefix(prefix))
                    self.counts[dtype] = max(self.counts[dtype], index + 1)
                    break

    def reg(self, reg: VReg) -> str:
        existing = self.names.get(reg.name)
        if existing is not None:
            return existing
        index = self.counts[reg.dtype]
        self.counts[reg.dtype] = index + 1
        physical = f"{reg.dtype.ptx_reg_prefix}{index}"
        self.names[reg.name] = physical
        return physical


def emit(kernel: KernelIR, *, allocation: str = "greedy") -> str:
    validate(kernel)
    alloc = _Alloc()
    if allocation == "greedy":
        from ptxlab.regalloc import greedy_map

        alloc.preload(greedy_map(kernel))
    elif allocation != "naive":
        raise EmitError(f"unknown allocation mode {allocation}")
    body: list[str] = []
    for block in kernel.blocks:
        body.append(f"{block.label}:")
        for instr in block.instrs:
            body.append(f"    {_render(instr, alloc)};")
        body.append(f"    {_render_term(block.term, alloc)};")
    return "\n".join(_header(kernel, alloc) + body + ["}", ""])


def _header(kernel: KernelIR, alloc: _Alloc) -> list[str]:
    lines = [
        f"// PTXLab {kernel.name}",
        f".version {format_version(kernel.version)}",
        f".target {kernel.target}",
        f".address_size {kernel.address_size}",
    ]
    if kernel.expect_block is not None:
        lines.append("// .ptxlab_block " + " ".join(str(dim) for dim in kernel.expect_block))
    lines.append(f".visible .entry {kernel.name} (")
    for index, param in enumerate(kernel.params):
        comma = "," if index + 1 < len(kernel.params) else ""
        kind = ".u64" if param.is_pointer else _scalar_kind(param.dtype)
        lines.append(f"    .param {kind} {param.name}{comma}")
    lines.append(")")
    lines.append("{")
    for dtype in (DType.PRED, DType.U32, DType.S32, DType.U64, DType.F32, DType.F16):
        count = alloc.counts[dtype]
        if count:
            lines.append(f"    .reg {dtype.ptx_reg_kind} {dtype.ptx_reg_prefix}<{count}>;")
    for item in kernel.shared:
        lines.append(f"    .shared .align 4 {_scalar_kind(item.dtype)} {item.name}[{item.length}];")
    return lines


def _render(instr: Instr, alloc: _Alloc) -> str:
    op = instr.op
    if op is Op.CONST:
        assert instr.dst is not None and instr.dtype is not None
        return f"mov.{_mov_suffix(instr.dtype)} {alloc.reg(instr.dst)}, {_immediate(instr.dtype, instr.args[0])}"
    if op is Op.MOV:
        assert isinstance(instr.dst, VReg) and isinstance(instr.args[0], VReg)
        suffix = _mov_suffix(instr.dst.dtype)
        return f"mov.{suffix} {alloc.reg(instr.dst)}, {alloc.reg(instr.args[0])}"
    if op in (Op.ADD, Op.SUB, Op.MUL):
        assert isinstance(instr.dst, VReg) and instr.dtype is not None
        lhs, rhs = instr.args
        assert isinstance(lhs, VReg) and isinstance(rhs, VReg)
        return (
            f"{_arith_opcode(op, instr.dtype)} {alloc.reg(instr.dst)}, "
            f"{alloc.reg(lhs)}, {alloc.reg(rhs)}"
        )
    if op is Op.MUL_WIDE:
        assert isinstance(instr.dst, VReg)
        lhs, rhs = instr.args
        assert isinstance(lhs, VReg) and isinstance(rhs, VReg)
        return f"mul.wide.u32 {alloc.reg(instr.dst)}, {alloc.reg(lhs)}, {alloc.reg(rhs)}"
    if op is Op.MAD_LO:
        assert isinstance(instr.dst, VReg) and instr.dtype is not None
        a, b, c = instr.args
        assert isinstance(a, VReg) and isinstance(b, VReg) and isinstance(c, VReg)
        return (
            f"mad.lo.{instr.dtype.value} {alloc.reg(instr.dst)}, "
            f"{alloc.reg(a)}, {alloc.reg(b)}, {alloc.reg(c)}"
        )
    if op is Op.SHR:
        assert isinstance(instr.dst, VReg)
        lhs, rhs = instr.args
        assert isinstance(lhs, VReg) and isinstance(rhs, VReg)
        return f"shr.u32 {alloc.reg(instr.dst)}, {alloc.reg(lhs)}, {alloc.reg(rhs)}"
    if op is Op.CMP:
        assert isinstance(instr.dst, VReg) and instr.dtype is not None and instr.cmp is not None
        lhs, rhs = instr.args
        assert isinstance(lhs, VReg) and isinstance(rhs, VReg)
        return (
            f"setp.{instr.cmp}.{instr.dtype.value} {alloc.reg(instr.dst)}, "
            f"{alloc.reg(lhs)}, {alloc.reg(rhs)}"
        )
    if op in (Op.AND_PRED, Op.OR_PRED):
        assert isinstance(instr.dst, VReg)
        lhs, rhs = instr.args
        assert isinstance(lhs, VReg) and isinstance(rhs, VReg)
        mnemonic = "and.pred" if op is Op.AND_PRED else "or.pred"
        return f"{mnemonic} {alloc.reg(instr.dst)}, {alloc.reg(lhs)}, {alloc.reg(rhs)}"
    if op is Op.NOT_PRED:
        assert isinstance(instr.dst, VReg) and isinstance(instr.args[0], VReg)
        return f"not.pred {alloc.reg(instr.dst)}, {alloc.reg(instr.args[0])}"
    if op in (Op.TID, Op.CTAID, Op.NTID):
        assert isinstance(instr.dst, VReg) and instr.dim is not None
        special = {"tid": "%tid", "ctaid": "%ctaid", "ntid": "%ntid"}[op.value]
        return f"mov.u32 {alloc.reg(instr.dst)}, {special}.{instr.dim}"
    if op is Op.LD_PARAM:
        assert isinstance(instr.dst, VReg) and instr.symbol is not None
        return f"ld.param.{_param_suffix(instr.dst.dtype)} {alloc.reg(instr.dst)}, [{instr.symbol}]"
    if op is Op.LD_GLOBAL:
        assert isinstance(instr.dst, VReg) and isinstance(instr.args[0], VReg)
        return f"ld.global.{instr.dst.dtype.value} {alloc.reg(instr.dst)}, [{alloc.reg(instr.args[0])}]"
    if op is Op.ST_GLOBAL:
        addr, value = instr.args
        assert isinstance(addr, VReg) and isinstance(value, VReg)
        return f"st.global.{value.dtype.value} [{alloc.reg(addr)}], {alloc.reg(value)}"
    if op is Op.SHARED_BASE:
        assert isinstance(instr.dst, VReg) and instr.symbol is not None
        return f"mov.u32 {alloc.reg(instr.dst)}, {instr.symbol}"
    if op is Op.LD_SHARED:
        assert isinstance(instr.dst, VReg) and isinstance(instr.args[0], VReg)
        return f"ld.shared.{instr.dst.dtype.value} {alloc.reg(instr.dst)}, [{alloc.reg(instr.args[0])}]"
    if op is Op.ST_SHARED:
        addr, value = instr.args
        assert isinstance(addr, VReg) and isinstance(value, VReg)
        return f"st.shared.{value.dtype.value} [{alloc.reg(addr)}], {alloc.reg(value)}"
    if op is Op.FMA:
        assert isinstance(instr.dst, VReg)
        a, b, c = instr.args
        assert isinstance(a, VReg) and isinstance(b, VReg) and isinstance(c, VReg)
        return f"fma.rn.f32 {alloc.reg(instr.dst)}, {alloc.reg(a)}, {alloc.reg(b)}, {alloc.reg(c)}"
    if op is Op.DIV:
        assert isinstance(instr.dst, VReg)
        left, right = instr.args
        assert isinstance(left, VReg) and isinstance(right, VReg)
        return f"div.rn.f32 {alloc.reg(instr.dst)}, {alloc.reg(left)}, {alloc.reg(right)}"
    if op is Op.SQRT:
        assert isinstance(instr.dst, VReg) and isinstance(instr.args[0], VReg)
        return f"sqrt.rn.f32 {alloc.reg(instr.dst)}, {alloc.reg(instr.args[0])}"
    if op is Op.MAX:
        assert isinstance(instr.dst, VReg)
        left, right = instr.args
        assert isinstance(left, VReg) and isinstance(right, VReg)
        return f"max.f32 {alloc.reg(instr.dst)}, {alloc.reg(left)}, {alloc.reg(right)}"
    if op is Op.EX2:
        assert isinstance(instr.dst, VReg) and isinstance(instr.args[0], VReg)
        return f"ex2.approx.ftz.f32 {alloc.reg(instr.dst)}, {alloc.reg(instr.args[0])}"
    if op is Op.CVT:
        assert isinstance(instr.dst, VReg) and isinstance(instr.args[0], VReg)
        src = instr.args[0]
        if instr.dst.dtype is DType.F32 and src.dtype is DType.F16:
            return f"cvt.f32.f16 {alloc.reg(instr.dst)}, {alloc.reg(src)}"
        if instr.dst.dtype is DType.F16 and src.dtype is DType.F32:
            return f"cvt.rn.f16.f32 {alloc.reg(instr.dst)}, {alloc.reg(src)}"
        raise EmitError("cvt only converts between f16 and f32")
    if op is Op.LD_GLOBAL_V4:
        assert isinstance(instr.dst, VReg) and isinstance(instr.args[0], VReg)
        regs = ", ".join(alloc.reg(reg) for reg in (instr.dst, *instr.extra_dsts))
        return f"ld.global.v4.f32 {{{regs}}}, [{alloc.reg(instr.args[0])}]"
    if op is Op.ST_GLOBAL_V4:
        addr, v0, v1, v2, v3 = instr.args
        assert isinstance(addr, VReg)
        values = ", ".join(alloc.reg(reg) for reg in (v0, v1, v2, v3))  # type: ignore[arg-type]
        return f"st.global.v4.f32 [{alloc.reg(addr)}], {{{values}}}"
    if op is Op.ATOM_ADD:
        assert isinstance(instr.dst, VReg)
        addr, value = instr.args
        assert isinstance(addr, VReg) and isinstance(value, VReg)
        return f"atom.add.{value.dtype.value} {alloc.reg(instr.dst)}, [{alloc.reg(addr)}], {alloc.reg(value)}"
    if op is Op.BAR:
        return f"bar.sync {instr.args[0]}"
    raise EmitError(f"cannot emit {op}")


def _render_term(term: object, alloc: _Alloc) -> str:
    if isinstance(term, Ret):
        return "ret"
    if isinstance(term, Cbr):
        # Two instructions would break the one-statement-per-line body. The
        # parser accepts a predicated bra and a fall-through bra as separate
        # lines, so the caller emits them as one logical terminator via a
        # newline. Keep a single line pair by using an explicit marker the
        # emitter splits below.
        return f"@{alloc.reg(term.pred)} bra {term.then_target};\n    bra {term.else_target}"
    return f"bra {term.target}"  # type: ignore[attr-defined]


def _arith_opcode(op: Op, dtype: DType) -> str:
    if dtype is DType.F32:
        if op is Op.MUL:
            return "mul.rn.f32"
        if op is Op.ADD:
            return "add.rn.f32"
        if op is Op.SUB:
            return "sub.rn.f32"
    if dtype is DType.U64 and op is Op.ADD:
        return "add.u64"
    if op is Op.MUL:
        return f"mul.lo.{dtype.value}"
    return f"{op.value}.{dtype.value}"


def _mov_suffix(dtype: DType) -> str:
    if dtype is DType.PRED:
        return "pred"
    return dtype.value


def _param_suffix(dtype: DType) -> str:
    if dtype is DType.U64:
        return "u64"
    return dtype.value


def _scalar_kind(dtype: DType) -> str:
    return f".{dtype.value}"


def _immediate(dtype: DType, value: object) -> str:
    if dtype is DType.F32:
        bits = int(np.asarray(np.float32(value)).view(np.uint32))
        return f"0f{bits:08x}"
    if isinstance(value, int):
        return str(value)
    raise EmitError(f"cannot encode immediate {value!r}")
