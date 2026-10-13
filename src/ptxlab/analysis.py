"""Static resource notes derived from typed IR.

These are classifications of the compiler's own address math. They are not
hardware performance counters and they are not simulator cycle estimates.
"""

from __future__ import annotations

from dataclasses import dataclass

from ptxlab.ir import Instr, KernelIR, Op, VReg


@dataclass(frozen=True)
class MemoryNote:
    space: str
    classification: str
    detail: str


def memory_notes(kernel: KernelIR) -> list[MemoryNote]:
    defs = _single_defs(kernel)
    notes = []
    for block in kernel.blocks:
        for instr in block.instrs:
            if instr.op not in (Op.LD_GLOBAL, Op.ST_GLOBAL, Op.LD_SHARED, Op.ST_SHARED):
                continue
            if not instr.args or not isinstance(instr.args[0], VReg):
                continue
            form = _address_form(defs, instr.args[0].name)
            if instr.op in (Op.LD_GLOBAL, Op.ST_GLOBAL):
                notes.append(MemoryNote("global", _global_class(form), form))
            else:
                notes.append(MemoryNote("shared", _bank_class(form), form))
    return notes


def arithmetic_intensity(kernel: KernelIR) -> dict[str, float | int]:
    """Count arithmetic and memory ops in the IR. This is not a roofline result."""

    arith_ops = {Op.ADD, Op.SUB, Op.MUL, Op.MAD_LO}
    arith = 0
    mem = 0
    for block in kernel.blocks:
        for instr in block.instrs:
            if instr.op in arith_ops:
                arith += 1
            if instr.op in (Op.LD_GLOBAL, Op.ST_GLOBAL, Op.LD_SHARED, Op.ST_SHARED):
                mem += 1
    return {
        "arithmetic_ops": arith,
        "memory_ops": mem,
        "ops_per_memory": (arith / mem) if mem else float("inf"),
    }


def _global_class(form: str) -> str:
    if form in {"tid.x", "gid.x"}:
        return "unit-stride"
    if form.startswith("stride:"):
        step = int(form.split(":", 1)[1])
        return "unit-stride" if step == 1 else "strided"
    return "unclassified"


def _bank_class(form: str) -> str:
    if form in {"tid.x", "gid.x"}:
        return "consecutive-words"
    if form.startswith("stride:"):
        step = int(form.split(":", 1)[1])
        if step % 32 == 0:
            return "same-bank"
        return "strided"
    return "unclassified"


def _address_form(defs: dict[str, Instr], name: str) -> str:
    seen: set[str] = set()
    while name in defs and name not in seen:
        seen.add(name)
        instr = defs[name]
        if instr.op is Op.MOV and isinstance(instr.args[0], VReg):
            name = instr.args[0].name
            continue
        if instr.op is Op.TID and instr.dim == "x":
            return "tid.x"
        if instr.op is Op.CTAID and instr.dim == "x":
            return "ctaid.x"
        if instr.op is Op.NTID and instr.dim == "x":
            return "ntid.x"
        if instr.op is Op.SHARED_BASE:
            return "shared-base"
        if instr.op is Op.MUL and instr.dtype is not None and instr.dtype.value == "u32":
            forms = [_form_of(defs, arg) for arg in instr.args if isinstance(arg, VReg)]
            if "ctaid.x" in forms and "ntid.x" in forms:
                return "ctaid*ntid"
            const = _const(defs, instr)
            if const == 4:
                other = next(arg for arg in instr.args if isinstance(arg, VReg) and not _is_const(defs, arg.name))
                return _form_of(defs, other)
            if const is not None:
                other = next(arg for arg in instr.args if isinstance(arg, VReg) and not _is_const(defs, arg.name))
                base = _form_of(defs, other)
                if base == "tid.x":
                    return f"stride:{const}"
        if instr.op is Op.ADD:
            forms = [_form_of(defs, arg) for arg in instr.args if isinstance(arg, VReg)]
            if "tid.x" in forms and "ctaid*ntid" in forms:
                return "gid.x"
            interesting = [form for form in forms if form not in {"unknown", "shared-base"}]
            if len(interesting) == 1:
                return interesting[0]
        if instr.op is Op.MUL_WIDE:
            index = next(
                (arg for arg in instr.args if isinstance(arg, VReg) and not _is_small_const(defs, arg.name)),
                None,
            )
            if index is not None:
                return _form_of(defs, index)
        break
    return "unknown"


def _form_of(defs: dict[str, Instr], reg: VReg) -> str:
    return _address_form(defs, reg.name)


def _const(defs: dict[str, Instr], instr: Instr) -> int | None:
    for arg in instr.args:
        if isinstance(arg, VReg) and _is_const(defs, arg.name):
            value = defs[arg.name].args[0]
            if isinstance(value, int):
                return value
    return None


def _is_const(defs: dict[str, Instr], name: str) -> bool:
    instr = defs.get(name)
    return instr is not None and instr.op is Op.CONST


def _is_small_const(defs: dict[str, Instr], name: str) -> bool:
    instr = defs.get(name)
    return bool(instr and instr.op is Op.CONST and isinstance(instr.args[0], int) and instr.args[0] <= 16)


def _single_defs(kernel: KernelIR) -> dict[str, Instr]:
    found: dict[str, Instr] = {}
    multi: set[str] = set()
    for block in kernel.blocks:
        for instr in block.instrs:
            if instr.dst is None:
                continue
            if instr.dst.name in found or instr.dst.name in multi:
                multi.add(instr.dst.name)
                found.pop(instr.dst.name, None)
            else:
                found[instr.dst.name] = instr
    return found


def _u32():
    from ptxlab.dtype import DType

    return DType.U32


def _u64():
    from ptxlab.dtype import DType

    return DType.U64
