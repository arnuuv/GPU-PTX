"""IR optimisation passes.

Every pass is semantics-preserving for the PTXLab interpreter and is gated by
``OptimizeConfig``. Loop unrolling and list scheduling are off by default:
they are tested for equivalence, and no GPU timing exists to claim they are
faster. Integer strength reduction rewrites ``x * 2`` as ``x + x`` and drops
add/sub of zero and mul by zero or one. ``f32`` identities that would change
signed zero or NaNs are not rewritten.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from ptxlab.dtype import DType
from ptxlab.ir import (
    Block,
    Br,
    Cbr,
    Instr,
    KernelIR,
    Op,
    VReg,
)
from ptxlab.validate import validate

_PURE = {
    Op.CONST,
    Op.MOV,
    Op.ADD,
    Op.SUB,
    Op.MUL,
    Op.MUL_WIDE,
    Op.MAD_LO,
    Op.SHR,
    Op.CMP,
    Op.AND_PRED,
    Op.OR_PRED,
    Op.NOT_PRED,
    Op.TID,
    Op.CTAID,
    Op.NTID,
    Op.LD_PARAM,
    Op.SHARED_BASE,
}
_SIDE = {Op.ST_GLOBAL, Op.ST_SHARED, Op.ST_GLOBAL_V4, Op.BAR, Op.ATOM_ADD}


@dataclass(frozen=True)
class OptimizeConfig:
    constant_fold: bool = True
    copy_prop: bool = True
    dce: bool = True
    cse: bool = True
    strength: bool = True
    licm: bool = True
    redundant_load: bool = True
    unroll: bool = False
    unroll_limit: int = 4
    schedule: bool = False

    @staticmethod
    def none() -> OptimizeConfig:
        return OptimizeConfig(
            constant_fold=False,
            copy_prop=False,
            dce=False,
            cse=False,
            strength=False,
            licm=False,
            redundant_load=False,
            unroll=False,
            schedule=False,
        )


def optimize(kernel: KernelIR, config: OptimizeConfig | None = None) -> tuple[KernelIR, list[str]]:
    """Return a rewritten kernel and a list of pass notes."""

    chosen = config or OptimizeConfig()
    current = _clone(kernel)
    notes: list[str] = []
    if chosen.unroll:
        current, count = _unroll(current, chosen.unroll_limit)
        notes.append(f"unroll: {count}")
    for _ in range(6):
        changed = 0
        if chosen.constant_fold:
            current, count = _fold(current)
            changed += count
        if chosen.strength:
            current, count = _strength(current)
            changed += count
        if chosen.cse:
            current, count = _cse(current)
            changed += count
        if chosen.redundant_load:
            current, count = _redundant_loads(current)
            changed += count
        if chosen.copy_prop:
            current, count = _copy_prop(current)
            changed += count
        if chosen.licm:
            current, count = _licm(current)
            changed += count
        if chosen.dce:
            current, count = _dce(current)
            changed += count
        if changed == 0:
            break
        notes.append(f"rewrite: {changed}")
    if chosen.schedule:
        current, count = _schedule(current)
        notes.append(f"schedule: {count}")
    validate(current)
    return current, notes


def _clone(kernel: KernelIR) -> KernelIR:
    return KernelIR(
        name=kernel.name,
        params=list(kernel.params),
        shared=list(kernel.shared),
        blocks=[Block(block.label, list(block.instrs), block.term) for block in kernel.blocks],
        version=kernel.version,
        target=kernel.target,
        address_size=kernel.address_size,
        expect_block=kernel.expect_block,
    )


def _single_defs(kernel: KernelIR) -> dict[str, Instr]:
    found: dict[str, Instr] = {}
    multi: set[str] = set()
    for block in kernel.blocks:
        for instr in block.instrs:
            if instr.dst is None:
                continue
            name = instr.dst.name
            if name in found or name in multi:
                multi.add(name)
                found.pop(name, None)
            else:
                found[name] = instr
    return found


def _constants(defs: dict[str, Instr]) -> dict[str, object]:
    values = {}
    for name, instr in defs.items():
        if instr.op is Op.CONST:
            values[name] = instr.args[0]
    return values


def _fold(kernel: KernelIR) -> tuple[KernelIR, int]:
    defs = _single_defs(kernel)
    constants = _constants(defs)
    count = 0
    for block in kernel.blocks:
        rewritten = []
        for instr in block.instrs:
            folded = _fold_one(instr, constants, defs)
            if folded is not None:
                instr = folded
                count += 1
            rewritten.append(instr)
        block.instrs = rewritten
    return kernel, count


def _fold_one(instr: Instr, constants: dict[str, object], defs: dict[str, Instr]) -> Instr | None:
    if instr.dst is None or instr.op not in (Op.ADD, Op.SUB, Op.MUL, Op.MUL_WIDE, Op.MAD_LO, Op.SHR):
        return None
    values = []
    for arg in instr.args:
        if not isinstance(arg, VReg) or arg.name not in constants:
            return None
        values.append(constants[arg.name])
    dtype = instr.dtype or instr.dst.dtype
    try:
        result = _eval(instr.op, dtype, values)
    except (TypeError, ValueError, OverflowError):
        return None
    if defs.get(instr.dst.name) is not instr and instr.dst.name in defs:
        return None
    return Instr(Op.CONST, instr.dst, (result,), dtype=dtype)


def _eval(op: Op, dtype: DType, values: list[object]) -> object:
    if op is Op.MUL_WIDE:
        return (int(values[0]) & 0xFFFFFFFF) * (int(values[1]) & 0xFFFFFFFF)
    if op is Op.SHR:
        return (int(values[0]) & 0xFFFFFFFF) >> (int(values[1]) & 31)
    if dtype is DType.F32:
        left, right = np.float32(values[0]), np.float32(values[1])
        if op is Op.ADD:
            return float(np.add(left, right, dtype=np.float32))
        if op is Op.SUB:
            return float(np.subtract(left, right, dtype=np.float32))
        if op is Op.MUL:
            return float(np.multiply(left, right, dtype=np.float32))
        raise ValueError(op)
    if dtype is DType.U64:
        if op is not Op.ADD:
            raise ValueError(op)
        return (int(values[0]) + int(values[1])) & 0xFFFFFFFFFFFFFFFF
    signed = dtype is DType.S32

    def norm(value: object) -> int:
        bits = int(value) & 0xFFFFFFFF
        if signed and bits >= 0x80000000:
            return bits - 0x100000000
        return bits

    if op is Op.MAD_LO:
        raw = norm(values[0]) * norm(values[1]) + norm(values[2])
    elif op is Op.ADD:
        raw = norm(values[0]) + norm(values[1])
    elif op is Op.SUB:
        raw = norm(values[0]) - norm(values[1])
    elif op is Op.MUL:
        raw = norm(values[0]) * norm(values[1])
    else:
        raise ValueError(op)
    return norm(raw)


def _strength(kernel: KernelIR) -> tuple[KernelIR, int]:
    constants = _constants(_single_defs(kernel))
    count = 0
    for block in kernel.blocks:
        rewritten = []
        for instr in block.instrs:
            replacement = _strength_one(instr, constants)
            if replacement is not None:
                instr = replacement
                count += 1
            rewritten.append(instr)
        block.instrs = rewritten
    return kernel, count


def _strength_one(instr: Instr, constants: dict[str, object]) -> Instr | None:
    if instr.dst is None or instr.dtype in (None, DType.F32, DType.PRED) or instr.op not in (Op.ADD, Op.SUB, Op.MUL):
        return None
    if len(instr.args) != 2 or not all(isinstance(arg, VReg) for arg in instr.args):
        return None
    left, right = instr.args
    assert isinstance(left, VReg) and isinstance(right, VReg)
    left_c = constants.get(left.name)
    right_c = constants.get(right.name)
    if instr.op is Op.ADD and (left_c == 0 or right_c == 0):
        source = right if left_c == 0 else left
        return Instr(Op.MOV, instr.dst, (source,))
    if instr.op is Op.SUB and right_c == 0:
        return Instr(Op.MOV, instr.dst, (left,))
    if instr.op is Op.MUL and (left_c == 1 or right_c == 1):
        source = right if left_c == 1 else left
        return Instr(Op.MOV, instr.dst, (source,))
    if instr.op is Op.MUL and (left_c == 0 or right_c == 0):
        return Instr(Op.CONST, instr.dst, (0,), dtype=instr.dtype)
    if instr.op is Op.MUL and (left_c == 2 or right_c == 2):
        source = right if left_c == 2 else left
        return Instr(Op.ADD, instr.dst, (source, source), dtype=instr.dtype)
    return None


def _cse(kernel: KernelIR) -> tuple[KernelIR, int]:
    defs = _single_defs(kernel)
    counts = _def_counts(kernel)
    count = 0
    for block in kernel.blocks:
        seen: dict[tuple, VReg] = {}
        rewritten = []
        for instr in block.instrs:
            key = _cse_key(instr, counts)
            if key is not None and instr.dst is not None and key in seen and seen[key].name in defs:
                instr = Instr(Op.MOV, instr.dst, (seen[key],))
                count += 1
            elif key is not None and instr.dst is not None and counts.get(instr.dst.name) == 1:
                seen[key] = instr.dst
            rewritten.append(instr)
        block.instrs = rewritten
    return kernel, count


def _cse_key(instr: Instr, counts: dict[str, int]) -> tuple | None:
    if instr.op not in _PURE or instr.op is Op.MOV or instr.dst is None:
        return None
    args = []
    for arg in instr.args:
        if isinstance(arg, VReg):
            if counts.get(arg.name) != 1:
                return None
            args.append(("v", arg.name))
        else:
            args.append(("c", arg))
    return (instr.op, tuple(args), instr.dtype, instr.cmp, instr.dim, instr.symbol)


def _redundant_loads(kernel: KernelIR) -> tuple[KernelIR, int]:
    counts = _def_counts(kernel)
    count = 0
    for block in kernel.blocks:
        known: dict[tuple[str, str], VReg] = {}
        rewritten = []
        for instr in block.instrs:
            if instr.op is Op.BAR or instr.op in (Op.ST_GLOBAL, Op.ST_SHARED):
                if instr.op is Op.BAR:
                    known.clear()
                elif instr.op is Op.ST_GLOBAL:
                    known = {key: value for key, value in known.items() if key[0] != "g"}
                else:
                    known = {key: value for key, value in known.items() if key[0] != "s"}
            space = "g" if instr.op is Op.LD_GLOBAL else "s" if instr.op is Op.LD_SHARED else None
            if space is not None and instr.dst is not None and instr.args and isinstance(instr.args[0], VReg):
                addr = instr.args[0]
                key = (space, addr.name)
                if counts.get(addr.name) == 1 and key in known and counts.get(known[key].name) == 1:
                    instr = Instr(Op.MOV, instr.dst, (known[key],))
                    count += 1
                else:
                    known[key] = instr.dst
            rewritten.append(instr)
        block.instrs = rewritten
    return kernel, count


def _copy_prop(kernel: KernelIR) -> tuple[KernelIR, int]:
    subst: dict[str, VReg] = {}
    for instr in _single_defs(kernel).values():
        if instr.op is Op.MOV and instr.dst is not None and isinstance(instr.args[0], VReg):
            source = instr.args[0]
            if source.name != instr.dst.name:
                subst[instr.dst.name] = source
    if not subst:
        return kernel, 0

    def follow(reg: VReg) -> VReg:
        seen = set()
        while reg.name in subst and reg.name not in seen:
            seen.add(reg.name)
            reg = subst[reg.name]
        return reg

    count = 0
    for block in kernel.blocks:
        rewritten = []
        for instr in block.instrs:
            args = tuple(follow(arg) if isinstance(arg, VReg) else arg for arg in instr.args)
            if args != instr.args:
                instr = replace(instr, args=args)
                count += 1
            rewritten.append(instr)
        block.instrs = rewritten
        if isinstance(block.term, Cbr):
            pred = follow(block.term.pred)
            if pred != block.term.pred:
                block.term = Cbr(pred, block.term.then_target, block.term.else_target)
                count += 1
    return kernel, count


def _dce(kernel: KernelIR) -> tuple[KernelIR, int]:
    removed = 0
    changed = True
    while changed:
        changed = False
        used: set[str] = set()
        for block in kernel.blocks:
            for instr in block.instrs:
                used.update(arg.name for arg in instr.args if isinstance(arg, VReg))
            if isinstance(block.term, Cbr):
                used.add(block.term.pred.name)
        for block in kernel.blocks:
            kept = []
            for instr in block.instrs:
                if (
                    instr.dst is not None
                    and instr.dst.name not in used
                    and instr.op not in _SIDE
                    and instr.op not in (Op.LD_GLOBAL, Op.LD_SHARED, Op.LD_GLOBAL_V4)
                ):
                    removed += 1
                    changed = True
                    continue
                kept.append(instr)
            block.instrs = kept
    return kernel, removed


def _licm(kernel: KernelIR) -> tuple[KernelIR, int]:
    succs, preds = _cfg(kernel)
    hoisted = 0
    for header, body, preheader in _loops(succs, preds):
        if preheader is None:
            continue
        pre = _block(kernel, preheader)
        if pre is None or not isinstance(pre.term, (Br, Cbr)):
            continue
        defined_in = set()
        for label in body:
            block = _block(kernel, label)
            if block is None:
                continue
            for instr in block.instrs:
                if instr.dst is not None:
                    defined_in.add(instr.dst.name)
        for label in sorted(body):
            block = _block(kernel, label)
            if block is None:
                continue
            stay = []
            move = []
            for instr in block.instrs:
                if _invariant(instr, defined_in):
                    move.append(instr)
                else:
                    stay.append(instr)
            if not move:
                continue
            block.instrs = stay
            pre.instrs.extend(move)
            hoisted += len(move)
            for instr in move:
                if instr.dst is not None:
                    defined_in.discard(instr.dst.name)
    return kernel, hoisted


def _invariant(instr: Instr, defined_in: set[str]) -> bool:
    if instr.op not in {Op.CONST, Op.LD_PARAM, Op.TID, Op.CTAID, Op.NTID, Op.SHARED_BASE}:
        return False
    return not any(isinstance(arg, VReg) and arg.name in defined_in for arg in instr.args)


def _schedule(kernel: KernelIR) -> tuple[KernelIR, int]:
    moved = 0
    for block in kernel.blocks:
        ordered = _list_schedule(block.instrs)
        if ordered != block.instrs:
            moved += 1
            block.instrs = ordered
    return kernel, moved


def _list_schedule(instrs: list[Instr]) -> list[Instr]:
    if len(instrs) < 2:
        return instrs
    produced: dict[str, int] = {}
    for index, instr in enumerate(instrs):
        if instr.dst is not None and instr.dst.name not in produced:
            produced[instr.dst.name] = index
    deps = {index: set() for index in range(len(instrs))}
    last_global = None
    last_shared = None
    last_bar = None
    for index, instr in enumerate(instrs):
        for arg in instr.args:
            if isinstance(arg, VReg) and arg.name in produced and produced[arg.name] != index:
                deps[index].add(produced[arg.name])
        if instr.op in (Op.LD_GLOBAL, Op.ST_GLOBAL):
            if last_global is not None:
                deps[index].add(last_global)
            last_global = index
        if instr.op in (Op.LD_SHARED, Op.ST_SHARED):
            if last_shared is not None:
                deps[index].add(last_shared)
            last_shared = index
        if instr.op is Op.BAR:
            deps[index].update(range(index))
            last_bar = index
        elif last_bar is not None:
            deps[index].add(last_bar)
    ready = [index for index, need in deps.items() if not need]
    scheduled: list[Instr] = []
    done: set[int] = set()
    while ready:
        ready.sort(key=lambda index: (0 if instrs[index].op in (Op.LD_GLOBAL, Op.LD_SHARED) else 1, index))
        pick = ready.pop(0)
        done.add(pick)
        scheduled.append(instrs[pick])
        for index, need in deps.items():
            if index in done or index in ready:
                continue
            if need <= done:
                ready.append(index)
    if len(scheduled) != len(instrs):
        return instrs
    return scheduled


def _unroll(kernel: KernelIR, limit: int) -> tuple[KernelIR, int]:
    succs, preds = _cfg(kernel)
    count = 0
    for header, body, preheader in list(_loops(succs, preds)):
        if preheader is None or len(body) != 2:
            continue
        header_block = _block(kernel, header)
        pre = _block(kernel, preheader)
        if header_block is None or pre is None or not isinstance(pre.term, Br):
            continue
        other = next(iter(body - {header}))
        body_block = _block(kernel, other)
        if body_block is None or not isinstance(body_block.term, Br) or body_block.term.target != header:
            continue
        plan = _unroll_plan(header_block, body_block, pre, limit)
        if plan is None:
            continue
        iv, start, step, trip, exit_label = plan
        straight: list[Instr] = []
        for iteration in range(trip):
            value = start + iteration * step
            const_reg = VReg(f"%unroll_{count}_{iteration}", DType.U32)
            straight.append(Instr(Op.CONST, const_reg, (value & 0xFFFFFFFF,), dtype=DType.U32))
            for instr in body_block.instrs:
                if instr.dst is not None and instr.dst.name == iv:
                    continue
                args = tuple(
                    const_reg if isinstance(arg, VReg) and arg.name == iv else arg for arg in instr.args
                )
                straight.append(replace(instr, args=args))
        label = f"L_unroll_{count}"
        pre.term = Br(label)
        kernel.blocks = [
            Block(label, straight, Br(exit_label)) if block.label == header else block
            for block in kernel.blocks
            if block.label != other
        ]
        count += 1
        break
    return kernel, count


def _unroll_plan(header: Block, body: Block, pre: Block, limit: int):
    if not isinstance(header.term, Cbr) or header.term.pred is None:
        return None
    cmp = next((instr for instr in header.instrs if instr.op is Op.CMP and instr.dst == header.term.pred), None)
    if cmp is None or cmp.cmp != "lt" or cmp.dtype is not DType.U32:
        return None
    if len(cmp.args) != 2 or not isinstance(cmp.args[0], VReg):
        return None
    iv = cmp.args[0].name
    end = _const_from_block(header, cmp.args[1])
    step = None
    for instr in body.instrs:
        if instr.op is Op.ADD and instr.dst is not None and instr.dst.name == iv and instr.dtype is DType.U32:
            if len(instr.args) == 2 and isinstance(instr.args[0], VReg) and instr.args[0].name == iv:
                step = _const_from_block(body, instr.args[1])
    if end is None or step is None or step <= 0:
        return None
    start = _assigned_const(pre, iv)
    if start is None or (end - start) % step != 0:
        return None
    trip = (end - start) // step
    if trip < 1 or trip > limit:
        return None
    exit_label = header.term.else_target
    return iv, start, step, trip, exit_label


def _assigned_const(block: Block, name: str) -> int | None:
    consts: dict[str, int] = {}
    value: int | None = None
    for instr in block.instrs:
        if instr.op is Op.CONST and instr.dst is not None and isinstance(instr.args[0], int):
            consts[instr.dst.name] = instr.args[0]
            if instr.dst.name == name:
                value = instr.args[0]
        elif (
            instr.op is Op.MOV
            and instr.dst is not None
            and instr.dst.name == name
            and isinstance(instr.args[0], VReg)
            and instr.args[0].name in consts
        ):
            value = consts[instr.args[0].name]
    return value


def _const_from_block(block: Block, reg: object) -> int | None:
    if not isinstance(reg, VReg):
        return None
    for instr in block.instrs:
        if instr.op is Op.CONST and instr.dst == reg and isinstance(instr.args[0], int):
            return instr.args[0]
    return None


def _def_counts(kernel: KernelIR) -> dict[str, int]:
    counts: dict[str, int] = {}
    for block in kernel.blocks:
        for instr in block.instrs:
            if instr.dst is not None:
                counts[instr.dst.name] = counts.get(instr.dst.name, 0) + 1
    return counts


def _cfg(kernel: KernelIR) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    succs: dict[str, list[str]] = {}
    preds: dict[str, list[str]] = {block.label: [] for block in kernel.blocks}
    for block in kernel.blocks:
        if isinstance(block.term, Cbr):
            edges = [block.term.then_target, block.term.else_target]
        elif isinstance(block.term, Br):
            edges = [block.term.target]
        else:
            edges = []
        succs[block.label] = edges
        for edge in edges:
            preds.setdefault(edge, []).append(block.label)
    return succs, preds


def _loops(succs: dict[str, list[str]], preds: dict[str, list[str]]):
    state = {label: 0 for label in succs}
    back: list[tuple[str, str]] = []

    def walk(label: str) -> None:
        state[label] = 1
        for succ in succs.get(label, []):
            if state.get(succ, 0) == 0:
                walk(succ)
            elif state.get(succ) == 1:
                back.append((label, succ))
        state[label] = 2

    if "entry" in state:
        walk("entry")
    found = []
    for tail, header in back:
        body = {header}
        stack = [tail]
        while stack:
            node = stack.pop()
            if node in body:
                continue
            body.add(node)
            stack.extend(preds.get(node, []))
        outside = [pred for pred in preds.get(header, []) if pred not in body]
        preheader = outside[0] if len(outside) == 1 else None
        found.append((header, body, preheader))
    found.sort(key=lambda item: len(item[1]))
    return found


def _block(kernel: KernelIR, label: str) -> Block | None:
    for block in kernel.blocks:
        if block.label == label:
            return block
    return None
