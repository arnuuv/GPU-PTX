"""Readable IR dump."""

from __future__ import annotations

from ptxlab.ir import Cbr, Instr, KernelIR, Op, Ret, VReg
from ptxlab.toolchain import format_version


def dump_ir(kernel: KernelIR) -> str:
    lines = [
        f"kernel {kernel.name}",
        f"  isa {format_version(kernel.version)}  target {kernel.target}  address_size {kernel.address_size}",
    ]
    if kernel.expect_block is not None:
        bx, by, bz = kernel.expect_block
        lines.append(f"  expect_block ({bx}, {by}, {bz})")
    for param in kernel.params:
        if param.is_pointer:
            shape = ", ".join(str(dim) for dim in param.shape)
            lines.append(f"  param {param.name}: ptr {param.dtype.value}[{shape}]")
        else:
            lines.append(f"  param {param.name}: {param.dtype.value}")
    for item in kernel.shared:
        shape = "x".join(str(dim) for dim in item.shape)
        lines.append(f"  shared {item.name}: {item.dtype.value}[{shape}]")
    for block in kernel.blocks:
        lines.append(f"{block.label}:")
        for instr in block.instrs:
            lines.append(f"  {format_instr(instr)}")
        lines.append(f"  {format_term(block.term)}")
    return "\n".join(lines) + "\n"


def format_instr(instr: Instr) -> str:
    dst = instr.dst.name if isinstance(instr.dst, VReg) else ""
    if instr.op is Op.CONST:
        return f"{dst} = const {instr.dtype.value} {instr.args[0]}"
    if instr.op is Op.MOV:
        return f"{dst} = mov {instr.args[0].name}"  # type: ignore[union-attr]
    if instr.op in (Op.ADD, Op.SUB, Op.MUL, Op.SHR):
        lhs, rhs = instr.args
        ty = instr.dtype.value if instr.dtype is not None else ""
        return f"{dst} = {instr.op.value}.{ty} {lhs.name}, {rhs.name}"  # type: ignore[union-attr]
    if instr.op is Op.MUL_WIDE:
        lhs, rhs = instr.args
        return f"{dst} = mul_wide.u32 {lhs.name}, {rhs.name}"  # type: ignore[union-attr]
    if instr.op is Op.MAD_LO:
        a, b, c = instr.args
        return f"{dst} = mad_lo.{instr.dtype.value} {a.name}, {b.name}, {c.name}"  # type: ignore[union-attr]
    if instr.op is Op.CMP:
        lhs, rhs = instr.args
        return f"{dst} = cmp.{instr.cmp}.{instr.dtype.value} {lhs.name}, {rhs.name}"  # type: ignore[union-attr]
    if instr.op in (Op.AND_PRED, Op.OR_PRED):
        lhs, rhs = instr.args
        return f"{dst} = {instr.op.value} {lhs.name}, {rhs.name}"  # type: ignore[union-attr]
    if instr.op is Op.NOT_PRED:
        return f"{dst} = not_pred {instr.args[0].name}"  # type: ignore[union-attr]
    if instr.op in (Op.TID, Op.CTAID, Op.NTID):
        return f"{dst} = {instr.op.value}.{instr.dim}"
    if instr.op is Op.LD_PARAM:
        return f"{dst} = ld_param {instr.symbol}"
    if instr.op is Op.LD_GLOBAL:
        return f"{dst} = ld_global {instr.args[0].name}"  # type: ignore[union-attr]
    if instr.op is Op.ST_GLOBAL:
        addr, value = instr.args
        return f"st_global {addr.name}, {value.name}"  # type: ignore[union-attr]
    if instr.op is Op.SHARED_BASE:
        return f"{dst} = shared_base {instr.symbol}"
    if instr.op is Op.LD_SHARED:
        return f"{dst} = ld_shared {instr.args[0].name}"  # type: ignore[union-attr]
    if instr.op is Op.ST_SHARED:
        addr, value = instr.args
        return f"st_shared {addr.name}, {value.name}"  # type: ignore[union-attr]
    if instr.op is Op.BAR:
        return f"bar.sync {instr.args[0]}"
    return f"{instr.op.value}"


def format_term(term: object) -> str:
    if isinstance(term, Ret):
        return "ret"
    if isinstance(term, Cbr):
        return f"cbr {term.pred.name}, {term.then_target}, {term.else_target}"
    if term is None:
        return "<missing terminator>"
    return f"br {term.target}"  # type: ignore[attr-defined]
