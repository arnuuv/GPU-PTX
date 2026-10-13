"""Structural and type checks for KernelIR."""

from __future__ import annotations

from ptxlab.dtype import DType
from ptxlab.errors import IrError
from ptxlab.ir import (
    CMP_CODES,
    DIMS,
    Block,
    Cbr,
    Instr,
    KernelIR,
    Op,
    Ret,
    VReg,
)
from ptxlab.toolchain import validate_toolchain

_IDENT = __import__("re").compile(r"[A-Za-z_][A-Za-z0-9_]*$")


def validate(kernel: KernelIR) -> None:
    validate_toolchain(kernel.version, kernel.target, kernel.address_size)
    if not _IDENT.match(kernel.name):
        raise IrError(f"invalid kernel name {kernel.name!r}")
    if not kernel.blocks:
        raise IrError("kernel has no blocks")

    labels = [block.label for block in kernel.blocks]
    if len(labels) != len(set(labels)):
        raise IrError("duplicate block label")
    label_set = set(labels)
    if kernel.blocks[0].label != "entry":
        raise IrError("the first block must be named 'entry'")

    params = kernel.param_map()
    if len(params) != len(kernel.params):
        raise IrError("duplicate parameter name")
    shared = kernel.shared_map()
    if len(shared) != len(kernel.shared):
        raise IrError("duplicate shared memory name")
    for name in list(params) + list(shared):
        if not _IDENT.match(name):
            raise IrError(f"invalid symbol name {name!r}")
    for param in kernel.params:
        if param.is_pointer and param.dtype not in (DType.F32, DType.F16, DType.S32, DType.U32):
            raise IrError(f"pointer parameter {param.name} has unsupported element type {param.dtype}")
        if not param.is_pointer and param.dtype not in (DType.F32, DType.S32, DType.U32):
            raise IrError(f"scalar parameter {param.name} has unsupported type {param.dtype}")
    for item in kernel.shared:
        if item.dtype not in (DType.F32, DType.S32, DType.U32):
            raise IrError(f"shared buffer {item.name} has unsupported element type")
        if item.length <= 0:
            raise IrError(f"shared buffer {item.name} must have a positive length")

    types: dict[str, DType] = {}
    defined: set[str] = set()
    for block in kernel.blocks:
        if block.term is None:
            raise IrError(f"block {block.label} is missing a terminator")
        for instr in block.instrs:
            _check_instr(instr, params, shared, types, defined)
        _check_term(block, label_set, types, defined)

    if not any(isinstance(block.term, Ret) for block in kernel.blocks):
        raise IrError("kernel has no ret terminator")
    _check_reachability(kernel)


def _bind(types: dict[str, DType], reg: VReg) -> None:
    previous = types.get(reg.name)
    if previous is not None and previous is not reg.dtype:
        raise IrError(f"{reg.name} is both {previous.value} and {reg.dtype.value}")
    types[reg.name] = reg.dtype


def _use(types: dict[str, DType], defined: set[str], reg: VReg) -> None:
    _bind(types, reg)
    if reg.name not in defined:
        raise IrError(f"use of {reg.name} before it is defined")


def _define(types: dict[str, DType], defined: set[str], reg: VReg) -> None:
    _bind(types, reg)
    defined.add(reg.name)


def _expect(reg: VReg, dtype: DType, what: str) -> None:
    if reg.dtype is not dtype:
        raise IrError(f"{what} expects {dtype.value}, got {reg.name}:{reg.dtype.value}")


def _check_instr(
    instr: Instr,
    params: dict,
    shared: dict,
    types: dict[str, DType],
    defined: set[str],
) -> None:
    op = instr.op
    if op is Op.CONST:
        if not isinstance(instr.dst, VReg) or instr.dtype is None:
            raise IrError("const needs a typed destination")
        if instr.dst.dtype is not instr.dtype:
            raise IrError("const destination type does not match the immediate")
        _check_immediate(instr.dtype, instr.args[0] if instr.args else None)
        _define(types, defined, instr.dst)
        return
    if op is Op.MOV:
        dst, src = _two_regs(instr)
        if dst.dtype is not src.dtype:
            raise IrError("mov types differ")
        _use(types, defined, src)
        _define(types, defined, dst)
        return
    if op in (Op.ADD, Op.SUB, Op.MUL):
        dst, lhs, rhs = _three_regs(instr)
        if dst.dtype is not instr.dtype or lhs.dtype is not instr.dtype or rhs.dtype is not instr.dtype:
            raise IrError(f"{op.value} operands must share {instr.dtype}")
        if instr.dtype not in (DType.F32, DType.S32, DType.U32, DType.U64):
            raise IrError(f"{op.value} does not support {instr.dtype}")
        if op is not Op.ADD and instr.dtype is DType.U64:
            raise IrError("only add is supported on u64 in this subset")
        _use(types, defined, lhs)
        _use(types, defined, rhs)
        _define(types, defined, dst)
        return
    if op is Op.MUL_WIDE:
        dst, lhs, rhs = _three_regs(instr)
        _expect(dst, DType.U64, "mul.wide")
        _expect(lhs, DType.U32, "mul.wide")
        _expect(rhs, DType.U32, "mul.wide")
        _use(types, defined, lhs)
        _use(types, defined, rhs)
        _define(types, defined, dst)
        return
    if op is Op.MAD_LO:
        if instr.dst is None or len(instr.args) != 3:
            raise IrError("mad.lo needs one destination and three sources")
        dst = instr.dst
        srcs = instr.args
        if not all(isinstance(item, VReg) for item in srcs):
            raise IrError("mad.lo sources must be registers")
        if instr.dtype not in (DType.S32, DType.U32):
            raise IrError("mad.lo supports s32 and u32")
        for reg in (dst, *srcs):
            _expect(reg, instr.dtype, "mad.lo")  # type: ignore[arg-type]
        for reg in srcs:
            _use(types, defined, reg)  # type: ignore[arg-type]
        _define(types, defined, dst)
        return
    if op is Op.SHR:
        dst, lhs, rhs = _three_regs(instr)
        for reg in (dst, lhs, rhs):
            _expect(reg, DType.U32, "shr")
        _use(types, defined, lhs)
        _use(types, defined, rhs)
        _define(types, defined, dst)
        return
    if op is Op.CMP:
        if instr.cmp not in CMP_CODES or instr.dtype not in (DType.S32, DType.U32):
            raise IrError("cmp supports eq/ne/lt/le/gt/ge on s32 and u32")
        dst, lhs, rhs = _three_regs(instr)
        _expect(dst, DType.PRED, "cmp")
        _expect(lhs, instr.dtype, "cmp")
        _expect(rhs, instr.dtype, "cmp")
        _use(types, defined, lhs)
        _use(types, defined, rhs)
        _define(types, defined, dst)
        return
    if op in (Op.AND_PRED, Op.OR_PRED):
        dst, lhs, rhs = _three_regs(instr)
        for reg in (dst, lhs, rhs):
            _expect(reg, DType.PRED, op.value)
        _use(types, defined, lhs)
        _use(types, defined, rhs)
        _define(types, defined, dst)
        return
    if op is Op.NOT_PRED:
        dst, src = _two_regs(instr)
        _expect(dst, DType.PRED, "not.pred")
        _expect(src, DType.PRED, "not.pred")
        _use(types, defined, src)
        _define(types, defined, dst)
        return
    if op in (Op.TID, Op.CTAID, Op.NTID):
        if not isinstance(instr.dst, VReg) or instr.dim not in DIMS:
            raise IrError(f"{op.value} needs a u32 destination and an x/y/z dimension")
        _expect(instr.dst, DType.U32, op.value)
        _define(types, defined, instr.dst)
        return
    if op is Op.LD_PARAM:
        if not isinstance(instr.dst, VReg) or not isinstance(instr.symbol, str):
            raise IrError("ld_param needs a destination and a parameter name")
        param = params.get(instr.symbol)
        if param is None:
            raise IrError(f"unknown parameter {instr.symbol}")
        expected = DType.U64 if param.is_pointer else param.dtype
        _expect(instr.dst, expected, "ld_param")
        _define(types, defined, instr.dst)
        return
    if op is Op.LD_GLOBAL:
        dst, addr = _two_regs(instr)
        _expect(addr, DType.U64, "ld.global")
        if dst.dtype not in (DType.F32, DType.F16, DType.S32, DType.U32):
            raise IrError("ld.global supports f32, f16, s32, and u32")
        _use(types, defined, addr)
        _define(types, defined, dst)
        return
    if op is Op.ST_GLOBAL:
        if instr.dst is not None or len(instr.args) != 2:
            raise IrError("st.global takes an address and a value")
        addr, value = instr.args
        if not isinstance(addr, VReg) or not isinstance(value, VReg):
            raise IrError("st.global operands must be registers")
        _expect(addr, DType.U64, "st.global")
        if value.dtype not in (DType.F32, DType.F16, DType.S32, DType.U32):
            raise IrError("st.global supports f32, f16, s32, and u32")
        _use(types, defined, addr)
        _use(types, defined, value)
        return
    if op is Op.SHARED_BASE:
        if not isinstance(instr.dst, VReg) or instr.symbol not in shared:
            raise IrError("shared_base needs a known shared symbol")
        _expect(instr.dst, DType.U32, "shared_base")
        _define(types, defined, instr.dst)
        return
    if op is Op.LD_SHARED:
        dst, addr = _two_regs(instr)
        _expect(addr, DType.U32, "ld.shared")
        if dst.dtype not in (DType.F32, DType.S32, DType.U32):
            raise IrError("ld.shared supports f32, s32, and u32")
        _use(types, defined, addr)
        _define(types, defined, dst)
        return
    if op is Op.ST_SHARED:
        if instr.dst is not None or len(instr.args) != 2:
            raise IrError("st.shared takes an address and a value")
        addr, value = instr.args
        if not isinstance(addr, VReg) or not isinstance(value, VReg):
            raise IrError("st.shared operands must be registers")
        _expect(addr, DType.U32, "st.shared")
        if value.dtype not in (DType.F32, DType.S32, DType.U32):
            raise IrError("st.shared supports f32, s32, and u32")
        _use(types, defined, addr)
        _use(types, defined, value)
        return
    if op is Op.FMA:
        if instr.dst is None or len(instr.args) != 3:
            raise IrError("fma needs a destination and three sources")
        srcs = instr.args
        if not all(isinstance(item, VReg) for item in srcs):
            raise IrError("fma sources must be registers")
        for reg in (instr.dst, *srcs):
            _expect(reg, DType.F32, "fma")  # type: ignore[arg-type]
        for reg in srcs:
            _use(types, defined, reg)  # type: ignore[arg-type]
        _define(types, defined, instr.dst)
        return
    if op in (Op.DIV, Op.MAX):
        dst, lhs, rhs = _three_regs(instr)
        for reg in (dst, lhs, rhs):
            _expect(reg, DType.F32, op.value)
        _use(types, defined, lhs)
        _use(types, defined, rhs)
        _define(types, defined, dst)
        return
    if op in (Op.SQRT, Op.EX2):
        dst, src = _two_regs(instr)
        _expect(dst, DType.F32, op.value)
        _expect(src, DType.F32, op.value)
        _use(types, defined, src)
        _define(types, defined, dst)
        return
    if op is Op.CVT:
        dst, src = _two_regs(instr)
        pair = (src.dtype, dst.dtype)
        if pair not in ((DType.F16, DType.F32), (DType.F32, DType.F16)):
            raise IrError("cvt converts f16 to f32 or f32 to f16")
        _use(types, defined, src)
        _define(types, defined, dst)
        return
    if op is Op.LD_GLOBAL_V4:
        if instr.dst is None or len(instr.extra_dsts) != 3 or len(instr.args) != 1:
            raise IrError("ld.global.v4 needs four f32 destinations and one address")
        addr = instr.args[0]
        if not isinstance(addr, VReg):
            raise IrError("ld.global.v4 address must be a register")
        _expect(addr, DType.U64, "ld.global.v4")
        for reg in (instr.dst, *instr.extra_dsts):
            _expect(reg, DType.F32, "ld.global.v4")
        _use(types, defined, addr)
        for reg in (instr.dst, *instr.extra_dsts):
            _define(types, defined, reg)
        return
    if op is Op.ST_GLOBAL_V4:
        if instr.dst is not None or len(instr.args) != 5:
            raise IrError("st.global.v4 takes an address and four values")
        addr, *values = instr.args
        if not isinstance(addr, VReg) or not all(isinstance(item, VReg) for item in values):
            raise IrError("st.global.v4 operands must be registers")
        _expect(addr, DType.U64, "st.global.v4")
        for reg in values:
            _expect(reg, DType.F32, "st.global.v4")  # type: ignore[arg-type]
        _use(types, defined, addr)
        for reg in values:
            _use(types, defined, reg)  # type: ignore[arg-type]
        return
    if op is Op.ATOM_ADD:
        if instr.dst is None or len(instr.args) != 2:
            raise IrError("atom.add needs a destination, an address, and a value")
        addr, value = instr.args
        if not isinstance(addr, VReg) or not isinstance(value, VReg):
            raise IrError("atom.add operands must be registers")
        _expect(addr, DType.U64, "atom.add")
        if value.dtype not in (DType.U32, DType.F32) or instr.dst.dtype is not value.dtype:
            raise IrError("atom.add supports u32 and f32")
        _use(types, defined, addr)
        _use(types, defined, value)
        _define(types, defined, instr.dst)
        return
    if op is Op.BAR:
        if instr.args != (0,) and instr.args != (1,) and not (
            len(instr.args) == 1 and isinstance(instr.args[0], int) and 0 <= instr.args[0] <= 15
        ):
            raise IrError("bar.sync barrier id must be an integer in 0..15")
        return
    raise IrError(f"unhandled op {op}")


def _check_term(block: Block, labels: set[str], types: dict[str, DType], defined: set[str]) -> None:
    term = block.term
    assert term is not None
    if isinstance(term, Ret):
        return
    if isinstance(term, Cbr):
        _expect(term.pred, DType.PRED, "branch")
        _use(types, defined, term.pred)
        targets = (term.then_target, term.else_target)
    else:
        targets = (term.target,)
    for target in targets:
        if target not in labels:
            raise IrError(f"block {block.label} branches to unknown label {target}")


def _check_immediate(dtype: DType, value: object) -> None:
    if dtype is DType.F32:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise IrError("f32 immediate must be a number")
        return
    if dtype in (DType.S32, DType.U32, DType.U64):
        if isinstance(value, bool) or not isinstance(value, int):
            raise IrError(f"{dtype.value} immediate must be an int")
        return
    raise IrError(f"cannot materialise an immediate {dtype.value}")


def _two_regs(instr: Instr) -> tuple[VReg, VReg]:
    if not isinstance(instr.dst, VReg) or len(instr.args) != 1 or not isinstance(instr.args[0], VReg):
        raise IrError(f"{instr.op.value} needs a destination and one register source")
    return instr.dst, instr.args[0]


def _three_regs(instr: Instr) -> tuple[VReg, VReg, VReg]:
    if (
        not isinstance(instr.dst, VReg)
        or len(instr.args) != 2
        or not isinstance(instr.args[0], VReg)
        or not isinstance(instr.args[1], VReg)
    ):
        raise IrError(f"{instr.op.value} needs a destination and two register sources")
    return instr.dst, instr.args[0], instr.args[1]


def _check_reachability(kernel: KernelIR) -> None:
    by_label = {block.label: block for block in kernel.blocks}
    seen: set[str] = set()
    stack = ["entry"]
    while stack:
        label = stack.pop()
        if label in seen:
            continue
        seen.add(label)
        term = by_label[label].term
        if isinstance(term, Ret):
            continue
        if isinstance(term, Cbr):
            stack.extend((term.then_target, term.else_target))
        else:
            stack.append(term.target)
    missing = [block.label for block in kernel.blocks if block.label not in seen]
    if missing:
        raise IrError(f"unreachable blocks: {', '.join(missing)}")
