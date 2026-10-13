"""CPU interpreter for the PTX subset emitted by PTXLab.

Functional semantics, deliberately narrower than an NVIDIA GPU:

* Only instructions produced by :mod:`ptxlab.emit` run. Anything else is rejected
  at parse time.
* Thread blocks run one after another, ``x`` fastest. That order is a simulator
  choice. Independent blocks may not be read as a hardware schedule.
* A warp is 32 consecutive linear thread ids
  (``tx + ty * blockDim.x + tz * blockDim.x * blockDim.y``).
* The warp issues one instruction to its active mask. A predicated instruction
  writes only the lanes whose predicate is true, and inactive lanes do not
  touch memory.
* A predicated ``bra`` whose active lanes disagree splits the mask. Each side
  runs until the immediate post-dominator of that branch, then the original
  mask continues there. Uniform branches do not split. This is post-dominator
  reconvergence, not a model of a hardware reconvergence scoreboard.
* ``bar.sync`` suspends the warp. The block continues when every warp that has
  not returned is waiting on the same barrier id. Reaching a barrier with a
  mask other than the warp's launch mask, or returning while another warp waits,
  is a divergent barrier and raises ``DivergentBarrierError``.
* Shared memory is zero-filled, private to the block, and byte-addressed.
* Global loads and stores are 4-byte aligned and must fall inside the host
  allocation passed for that pointer. The host is little-endian.
* Registers start at 0. PTX leaves uninitialised registers undefined; zero is a
  simulator definition.
* ``u32`` and ``u64`` wrap modulo 2**32 and 2**64. ``s32`` wraps modulo 2**32
  and is then interpreted as signed. ``f32`` arithmetic is IEEE-754 binary32
  with round-to-nearest-even. ``shr.u32`` masks the shift by 31.
* Blocks and lanes inside one issued instruction are sequential. The simulator
  does not define a race outcome.
"""

from __future__ import annotations

import math
import struct
import sys
from collections import Counter
from dataclasses import dataclass, field

import numpy as np

from ptxlab.errors import DivergentBarrierError, LaunchError, MemoryFault, SimulationError
from ptxlab.parse import Program, SInst

_WARP = 32
_MAX_BLOCK_THREADS = 1024
_STEP_LIMIT = 5_000_000
_FMT = {"f32": "<f", "s32": "<i", "u32": "<I"}


@dataclass
class MemAccess:
    op: str
    block: tuple[int, int, int]
    warp: int
    pc: int
    addrs: tuple[int, ...]


@dataclass
class TraceEvent:
    block: tuple[int, int, int]
    warp: int
    pc: int
    op: str
    active_mask: int


@dataclass
class SimResult:
    inst_counts: dict[str, int]
    divergent_branches: int
    mem_accesses: list[MemAccess]
    steps: int
    trace: list[TraceEvent] = field(default_factory=list)

    @property
    def instructions(self) -> int:
        return sum(self.inst_counts.values())


@dataclass
class _Frame:
    pc: int
    mask: int
    join: int | None


class _GlobalMem:
    def __init__(self) -> None:
        self.regions: list[tuple[int, bytearray]] = []
        self._next = 0x100000000

    def alloc(self, data: bytes) -> tuple[int, bytearray]:
        self._next = (self._next + 255) & ~255
        base = self._next
        buf = bytearray(data)
        self.regions.append((base, buf))
        self._next = (base + max(len(buf), 1) + 255) & ~255
        return base, buf

    def read(self, addr: int, size: int) -> bytes:
        buf, offset = self._region(addr, size)
        return bytes(buf[offset : offset + size])

    def write(self, addr: int, data: bytes) -> None:
        buf, offset = self._region(addr, len(data))
        buf[offset : offset + len(data)] = data

    def _region(self, addr: int, size: int) -> tuple[bytearray, int]:
        for base, buf in self.regions:
            if base <= addr and addr + size <= base + len(buf):
                return buf, addr - base
        raise MemoryFault(f"address {addr:#x} size {size} is outside allocated global memory")


class _Warp:
    def __init__(self, warp_id: int, launch_mask: int, block: tuple[int, int, int], block_dim: tuple[int, int, int]) -> None:
        self.warp_id = warp_id
        self.launch_mask = launch_mask
        self.block = block
        self.block_dim = block_dim
        self.stack: list[_Frame] = [_Frame(0, launch_mask, None)]
        self.values: dict[str, list[object]] = {}
        self.finished = False
        self.blocked: int | None = None
        self.divergent = 0

    def get(self, name: str, lane: int) -> object:
        return self.values.setdefault(name, [0] * _WARP)[lane]

    def set(self, name: str, lane: int, value: object) -> None:
        self.values.setdefault(name, [0] * _WARP)[lane] = value


def simulate(
    program: Program,
    *,
    grid: tuple[int, ...] | list[int],
    block: tuple[int, ...] | list[int],
    args: dict[str, object],
    trace: bool = False,
) -> SimResult:
    if sys.byteorder != "little":
        raise SimulationError("the functional simulator requires a little-endian host")
    grid3 = _dims(grid, "grid")
    block3 = _dims(block, "block")
    threads = block3[0] * block3[1] * block3[2]
    if threads > _MAX_BLOCK_THREADS:
        raise LaunchError(f"block has {threads} threads; the simulator limit is {_MAX_BLOCK_THREADS}")
    if program.expect_block is not None and block3 != program.expect_block:
        raise LaunchError(f"kernel expects block {program.expect_block}, launch is {block3}")
    mem = _GlobalMem()
    params, outputs = _bind_args(program, args, mem)
    ipdom = _immediate_postdominators(program)
    counts: Counter[str] = Counter()
    accesses: list[MemAccess] = []
    events: list[TraceEvent] = []
    divergent = 0
    steps = 0
    shared_bases, shared_bytes = _shared_layout(program)
    for bz in range(grid3[2]):
        for by in range(grid3[1]):
            for bx in range(grid3[0]):
                block_steps, block_div = _run_block(
                    program,
                    (bx, by, bz),
                    block3,
                    params,
                    mem,
                    shared_bases,
                    shared_bytes,
                    ipdom,
                    counts,
                    accesses,
                    events if trace else None,
                )
                steps += block_steps
                divergent += block_div
    for array, flat, base, buf in outputs:
        decoded = np.frombuffer(bytes(buf), dtype=flat.dtype).reshape(flat.shape).copy()
        array[...] = decoded.reshape(array.shape)
    return SimResult(dict(counts), divergent, accesses, steps, events)


def _run_block(
    program: Program,
    block_idx: tuple[int, int, int],
    block_dim: tuple[int, int, int],
    params: dict[str, object],
    mem: _GlobalMem,
    shared_bases: dict[str, int],
    shared_bytes: int,
    ipdom: dict[int, int | str],
    counts: Counter[str],
    accesses: list[MemAccess],
    events: list[TraceEvent] | None,
) -> tuple[int, int]:
    threads = block_dim[0] * block_dim[1] * block_dim[2]
    warps = []
    for warp_id in range((threads + _WARP - 1) // _WARP):
        live = min(_WARP, threads - warp_id * _WARP)
        warps.append(_Warp(warp_id, (1 << live) - 1, block_idx, block_dim))
    shared = bytearray(shared_bytes)
    steps = 0
    while any(not warp.finished for warp in warps):
        if steps > _STEP_LIMIT:
            raise SimulationError("exceeded the per-block instruction limit")
        progress = False
        for warp in warps:
            if warp.finished or warp.blocked is not None:
                continue
            _step(program, warp, mem, shared, shared_bases, params, ipdom, counts, accesses, events)
            steps += 1
            progress = True
        waiting = [warp for warp in warps if warp.blocked is not None]
        running = [warp for warp in warps if not warp.finished]
        if waiting and len(waiting) == len(running):
            ids = {warp.blocked for warp in waiting}
            if len(ids) != 1:
                raise DivergentBarrierError(f"block {block_idx} has warps waiting on barriers {sorted(ids)}")
            for warp in waiting:
                warp.blocked = None
                warp.stack[-1].pc += 1
            progress = True
        elif waiting and not progress:
            raise DivergentBarrierError(f"block {block_idx} reached a barrier with a missing warp")
        if not progress:
            raise SimulationError(f"block {block_idx} made no progress")
    return steps, sum(warp.divergent for warp in warps)


def _step(
    program: Program,
    warp: _Warp,
    mem: _GlobalMem,
    shared: bytearray,
    shared_bases: dict[str, int],
    params: dict[str, object],
    ipdom: dict[int, int | str],
    counts: Counter[str],
    accesses: list[MemAccess],
    events: list[TraceEvent] | None,
) -> None:
    if not warp.stack:
        warp.finished = True
        return
    frame = warp.stack[-1]
    if frame.mask == 0 or frame.pc >= len(program.insts) or (frame.join is not None and frame.pc == frame.join):
        warp.stack.pop()
        if not warp.stack:
            warp.finished = True
        return
    inst = program.insts[frame.pc]
    if inst.op == "bra":
        _branch(program, warp, frame, inst, ipdom)
        counts[inst.op] += 1
        if events is not None:
            events.append(TraceEvent(warp.block, warp.warp_id, frame.pc, inst.op, frame.mask))
        return
    if inst.op == "ret":
        counts[inst.op] += 1
        warp.stack.pop()
        if not warp.stack:
            warp.finished = True
        return
    if inst.op == "bar.sync":
        if frame.mask != warp.launch_mask or inst.pred is not None:
            raise DivergentBarrierError(
                f"block {warp.block} warp {warp.warp_id} hit bar.sync with a split mask"
            )
        counts[inst.op] += 1
        warp.blocked = int(inst.srcs[0])
        return
    lanes = [lane for lane in range(_WARP) if _lane_on(inst, warp, frame.mask, lane)]
    addrs: list[int] = []
    for lane in lanes:
        addr = _exec(program, inst, warp, lane, mem, shared, shared_bases, params)
        if addr is not None:
            addrs.append(addr)
    if addrs:
        accesses.append(MemAccess(inst.op, warp.block, warp.warp_id, frame.pc, tuple(addrs)))
    counts[inst.op] += 1
    if events is not None:
        events.append(TraceEvent(warp.block, warp.warp_id, frame.pc, inst.op, _mask_of(lanes)))
    frame.pc += 1


def _branch(program: Program, warp: _Warp, frame: _Frame, inst: SInst, ipdom: dict[int, int | str]) -> None:
    target = program.labels[str(inst.srcs[0])]
    taking = 0
    falling = 0
    for lane in range(_WARP):
        bit = 1 << lane
        if frame.mask & bit == 0:
            continue
        if inst.pred is None or _pred_true(inst, warp, lane):
            taking |= bit
        else:
            falling |= bit
    if taking and falling:
        join = ipdom[frame.pc]
        join_pc = len(program.insts) if join == "exit" else int(join)
        parent_join = frame.join
        parent_mask = frame.mask
        warp.stack.pop()
        warp.stack.append(_Frame(join_pc, parent_mask, parent_join))
        if falling:
            warp.stack.append(_Frame(frame.pc + 1, falling, join_pc))
        if taking:
            warp.stack.append(_Frame(target, taking, join_pc))
        warp.divergent += 1
        return
    if taking:
        frame.pc = target
    else:
        frame.pc += 1


def _exec(
    program: Program,
    inst: SInst,
    warp: _Warp,
    lane: int,
    mem: _GlobalMem,
    shared: bytearray,
    shared_bases: dict[str, int],
    params: dict[str, object],
) -> int | None:
    op = inst.op

    def src(index: int) -> object:
        return _value(inst.srcs[index], warp, lane, shared_bases)

    def write(value: object) -> None:
        if inst.dst is None:
            raise SimulationError(f"{op} has no destination")
        warp.set(inst.dst, lane, value)

    if op == "mov.u32":
        write(_as_u32(src(0)))
    elif op == "mov.u64":
        write(_as_u64(src(0)))
    elif op == "mov.s32":
        write(_as_s32(src(0)))
    elif op == "mov.f32":
        write(float(np.float32(src(0))))
    elif op == "mov.pred":
        write(bool(src(0)))
    elif op == "add.u32":
        write(_as_u32(_as_u32(src(0)) + _as_u32(src(1))))
    elif op == "add.s32":
        write(_as_s32(_as_s32(src(0)) + _as_s32(src(1))))
    elif op == "add.u64":
        write(_as_u64(_as_u64(src(0)) + _as_u64(src(1))))
    elif op == "add.rn.f32":
        write(_fadd(src(0), src(1)))
    elif op == "sub.u32":
        write(_as_u32(_as_u32(src(0)) - _as_u32(src(1))))
    elif op == "sub.s32":
        write(_as_s32(_as_s32(src(0)) - _as_s32(src(1))))
    elif op == "sub.rn.f32":
        write(_fsub(src(0), src(1)))
    elif op == "mul.lo.u32":
        write(_as_u32(_as_u32(src(0)) * _as_u32(src(1))))
    elif op == "mul.lo.s32":
        write(_as_s32(_as_s32(src(0)) * _as_s32(src(1))))
    elif op == "mul.rn.f32":
        write(_fmul(src(0), src(1)))
    elif op == "mul.wide.u32":
        write(_as_u64(_as_u32(src(0)) * _as_u32(src(1))))
    elif op == "mad.lo.u32":
        write(_as_u32(_as_u32(src(0)) * _as_u32(src(1)) + _as_u32(src(2))))
    elif op == "mad.lo.s32":
        write(_as_s32(_as_s32(src(0)) * _as_s32(src(1)) + _as_s32(src(2))))
    elif op == "shr.u32":
        write(_as_u32(_as_u32(src(0)) >> (_as_u32(src(1)) & 31)))
    elif op.startswith("setp."):
        _, cmp, ty = op.split(".")
        left = _as_u32(src(0)) if ty == "u32" else _as_s32(src(0))
        right = _as_u32(src(1)) if ty == "u32" else _as_s32(src(1))
        write(_compare(cmp, left, right))
    elif op == "and.pred":
        write(bool(src(0)) and bool(src(1)))
    elif op == "or.pred":
        write(bool(src(0)) or bool(src(1)))
    elif op == "not.pred":
        write(not bool(src(0)))
    elif op == "fma.rn.f32":
        write(_fma(src(0), src(1), src(2)))
    elif op == "div.rn.f32":
        write(_fdiv(src(0), src(1)))
    elif op == "sqrt.rn.f32":
        write(_fsqrt(src(0)))
    elif op == "max.f32":
        write(_fmax(src(0), src(1)))
    elif op == "ex2.approx.ftz.f32":
        write(_ex2(src(0)))
    elif op == "cvt.f32.f16":
        write(float(np.float32(np.float16(src(0)))))
    elif op == "cvt.rn.f16.f32":
        write(float(np.float16(np.float32(src(0)))))
    elif op == "ld.global.v4.f32":
        addr = _as_u64(src(0))
        if addr % 16:
            raise MemoryFault(f"misaligned v4 address {addr:#x}")
        raw = mem.read(addr, 16)
        values = np.frombuffer(raw, dtype=np.float32).copy()
        targets = inst.dsts or (inst.dst,)
        for reg, value in zip(targets, values, strict=True):
            warp.set(reg, lane, float(value))
        return addr
    elif op == "st.global.v4.f32":
        addr = _as_u64(src(0))
        if addr % 16:
            raise MemoryFault(f"misaligned v4 address {addr:#x}")
        payload = np.array([np.float32(src(i + 1)) for i in range(4)], dtype=np.float32)
        mem.write(addr, payload.tobytes())
        return addr
    elif op.startswith("atom.add."):
        kind = op.rsplit(".", 1)[1]
        addr = _as_u64(src(0))
        old = _load(mem.read, addr, kind)
        updated = _as_u32(old) + _as_u32(src(1)) if kind == "u32" else _fadd(old, src(1))
        _store(mem.write, addr, updated, kind)
        write(old)
        return addr
    elif op.startswith("ld.param."):
        write(params[str(inst.srcs[0])])
    elif op.startswith("ld.global."):
        addr = _as_u64(src(0))
        write(_load(mem.read, addr, op.rsplit(".", 1)[1]))
        return addr
    elif op.startswith("st.global."):
        addr = _as_u64(src(0))
        _store(mem.write, addr, src(1), op.rsplit(".", 1)[1])
        return addr
    elif op.startswith("ld.shared."):
        addr = _as_u32(src(0))
        write(_load(lambda a, n: _shared_read(shared, a, n), addr, op.rsplit(".", 1)[1]))
        return addr
    elif op.startswith("st.shared."):
        addr = _as_u32(src(0))
        _store(lambda a, data: _shared_write(shared, a, data), addr, src(1), op.rsplit(".", 1)[1])
        return addr
    else:
        raise SimulationError(f"no functional implementation for {op}")
    return None


def _value(token: object, warp: _Warp, lane: int, shared_bases: dict[str, int]) -> object:
    if isinstance(token, (int, float)) and not isinstance(token, bool):
        return token
    if not isinstance(token, str):
        raise SimulationError(f"bad operand {token!r}")
    if token[:5] in ("%tid.", "%nti") or token.startswith("%ctaid.") or token.startswith("%ntid."):
        return _special(token, warp, lane)
    if token.startswith("%"):
        return warp.get(token, lane)
    if token in shared_bases:
        return shared_bases[token]
    raise SimulationError(f"unknown operand {token}")


def _special(token: str, warp: _Warp, lane: int) -> int:
    linear = warp.warp_id * _WARP + lane
    tx, ty, tz = _unpack(linear, warp.block_dim)
    table = {
        "%tid.x": tx,
        "%tid.y": ty,
        "%tid.z": tz,
        "%ntid.x": warp.block_dim[0],
        "%ntid.y": warp.block_dim[1],
        "%ntid.z": warp.block_dim[2],
        "%ctaid.x": warp.block[0],
        "%ctaid.y": warp.block[1],
        "%ctaid.z": warp.block[2],
    }
    if token not in table:
        raise SimulationError(f"unsupported special register {token}")
    return table[token]


def _lane_on(inst: SInst, warp: _Warp, mask: int, lane: int) -> bool:
    if mask & (1 << lane) == 0:
        return False
    if inst.pred is None:
        return True
    return _pred_true(inst, warp, lane)


def _pred_true(inst: SInst, warp: _Warp, lane: int) -> bool:
    assert inst.pred is not None
    value = bool(warp.get(inst.pred, lane))
    return (not value) if inst.pred_not else value


def _bind_args(program: Program, args: dict[str, object], mem: _GlobalMem):
    params: dict[str, object] = {}
    outputs: list[tuple[np.ndarray, np.ndarray, int, bytearray]] = []
    expected = {param.name for param in program.params}
    missing = expected - set(args)
    extra = set(args) - expected
    if missing or extra:
        raise LaunchError(f"argument names do not match the kernel (missing {sorted(missing)}, extra {sorted(extra)})")
    for param in program.params:
        value = args[param.name]
        if param.kind == "u64":
            if not isinstance(value, np.ndarray):
                raise LaunchError(f"pointer parameter {param.name} expects a numpy array")
            if value.dtype not in (np.float32, np.float16, np.int32, np.uint32):
                raise LaunchError(f"{param.name} uses {value.dtype}, which is outside f32/f16/s32/u32")
            flat = np.ascontiguousarray(value)
            base, buf = mem.alloc(flat.tobytes())
            params[param.name] = base
            outputs.append((value, flat, base, buf))
        elif param.kind == "u32":
            params[param.name] = _as_u32(value)
        elif param.kind == "s32":
            params[param.name] = _as_s32(value)
        elif param.kind == "f32":
            params[param.name] = float(np.float32(value))
        else:
            raise LaunchError(f"unsupported parameter kind {param.kind}")
    return params, outputs


def _shared_layout(program: Program) -> tuple[dict[str, int], int]:
    bases: dict[str, int] = {}
    offset = 0
    for item in program.shared:
        bases[item.name] = offset
        offset += item.nbytes
    return bases, offset


def _immediate_postdominators(program: Program) -> dict[int, int | str]:
    count = len(program.insts)
    succs: dict[int | str, list[int | str]] = {}
    for pc, inst in enumerate(program.insts):
        if inst.op == "ret":
            succs[pc] = ["exit"]
        elif inst.op == "bra" and inst.pred is None:
            succs[pc] = [program.labels[str(inst.srcs[0])]]
        elif inst.op == "bra":
            succs[pc] = [program.labels[str(inst.srcs[0])], pc + 1]
        else:
            succs[pc] = [pc + 1 if pc + 1 < count else "exit"]
    succs["exit"] = []
    nodes: list[int | str] = list(range(count)) + ["exit"]
    post = {node: set(nodes) for node in nodes}
    post["exit"] = {"exit"}
    changed = True
    guard = 0
    while changed:
        changed = False
        guard += 1
        if guard > (count + 2) ** 2:
            raise SimulationError("post-dominator analysis did not converge")
        for pc in range(count):
            inter: set[int | str] | None = None
            for succ in succs[pc]:
                inter = set(post[succ]) if inter is None else inter & post[succ]
            new = {pc} | (inter or set())
            if new != post[pc]:
                post[pc] = new
                changed = True
    ipdom: dict[int, int | str] = {}
    for pc in range(count):
        candidates = [item for item in post[pc] if item != pc]
        if not candidates:
            ipdom[pc] = "exit"
            continue
        best = max(candidates, key=lambda item: len(post[item]))
        if any(item != best and item not in post[best] for item in candidates):
            raise SimulationError(f"no immediate post-dominator at pc {pc}")
        ipdom[pc] = best
    return ipdom


def _dims(values: tuple[int, ...] | list[int], what: str) -> tuple[int, int, int]:
    if not 1 <= len(values) <= 3:
        raise LaunchError(f"{what} must have 1 to 3 dimensions")
    dims = [int(values[0]), 1, 1]
    if len(values) > 1:
        dims[1] = int(values[1])
    if len(values) > 2:
        dims[2] = int(values[2])
    if any(dim <= 0 for dim in dims):
        raise LaunchError(f"{what} dimensions must be positive, got {tuple(dims)}")
    return dims[0], dims[1], dims[2]


def _unpack(linear: int, block: tuple[int, int, int]) -> tuple[int, int, int]:
    tx = linear % block[0]
    ty = (linear // block[0]) % block[1]
    tz = linear // (block[0] * block[1])
    return tx, ty, tz


def _mask_of(lanes: list[int]) -> int:
    mask = 0
    for lane in lanes:
        mask |= 1 << lane
    return mask


def _load(read, addr: int, kind: str) -> object:
    size = 2 if kind == "f16" else 4
    if addr % size:
        raise MemoryFault(f"misaligned {kind} address {addr:#x}")
    data = read(addr, size)
    if kind == "f16":
        return float(np.frombuffer(data, dtype=np.float16)[0])
    return struct.unpack(_FMT[kind], data)[0]


def _store(write, addr: int, value: object, kind: str) -> None:
    size = 2 if kind == "f16" else 4
    if addr % size:
        raise MemoryFault(f"misaligned {kind} address {addr:#x}")
    if kind == "f16":
        write(addr, np.float16(value).tobytes())
        return
    if kind == "f32":
        packed = struct.pack(_FMT[kind], float(np.float32(value)))
    elif kind == "s32":
        packed = struct.pack(_FMT[kind], _as_s32(value))
    else:
        packed = struct.pack(_FMT[kind], _as_u32(value))
    write(addr, packed)


def _shared_read(shared: bytearray, addr: int, size: int) -> bytes:
    if addr < 0 or addr + size > len(shared):
        raise MemoryFault(f"shared address {addr:#x} is outside the {len(shared)}-byte allocation")
    return bytes(shared[addr : addr + size])


def _shared_write(shared: bytearray, addr: int, data: bytes) -> None:
    if addr < 0 or addr + len(data) > len(shared):
        raise MemoryFault(f"shared address {addr:#x} is outside the {len(shared)}-byte allocation")
    shared[addr : addr + len(data)] = data


def _as_u32(value: object) -> int:
    return int(value) & 0xFFFFFFFF  # type: ignore[arg-type]


def _as_u64(value: object) -> int:
    return int(value) & 0xFFFFFFFFFFFFFFFF  # type: ignore[arg-type]


def _as_s32(value: object) -> int:
    bits = int(value) & 0xFFFFFFFF  # type: ignore[arg-type]
    return bits - 0x100000000 if bits >= 0x80000000 else bits


def _fadd(left: object, right: object) -> float:
    return float(np.add(np.float32(left), np.float32(right), dtype=np.float32))


def _fsub(left: object, right: object) -> float:
    return float(np.subtract(np.float32(left), np.float32(right), dtype=np.float32))


def _fmul(left: object, right: object) -> float:
    return float(np.multiply(np.float32(left), np.float32(right), dtype=np.float32))


def _fma(left: object, mid: object, right: object) -> float:
    return float(np.float32(math.fma(float(np.float32(left)), float(np.float32(mid)), float(np.float32(right)))))


def _fdiv(left: object, right: object) -> float:
    return float(np.divide(np.float32(left), np.float32(right), dtype=np.float32))


def _fsqrt(value: object) -> float:
    return float(np.sqrt(np.float32(value)))


def _fmax(left: object, right: object) -> float:
    a = np.float32(left)
    b = np.float32(right)
    if np.isnan(a):
        return float(b)
    if np.isnan(b):
        return float(a)
    return float(np.maximum(a, b))


def _ex2(value: object) -> float:
    """Simulator stand-in for ``ex2.approx.ftz.f32``.

    Subnormal inputs and outputs flush to zero. The finite result is
    ``exp2`` rounded to binary32. That is not NVIDIA's special-function
    approximation, so device results can differ.
    """

    number = np.float32(value)
    tiny = np.finfo(np.float32).tiny
    if number != 0 and abs(float(number)) < tiny:
        number = np.float32(0)
    result = np.float32(np.exp2(number))
    if result != 0 and abs(float(result)) < tiny:
        result = np.float32(0)
    return float(result)


def _compare(cmp: str, left: int, right: int) -> bool:
    return {
        "eq": left == right,
        "ne": left != right,
        "lt": left < right,
        "le": left <= right,
        "gt": left > right,
        "ge": left >= right,
    }[cmp]
