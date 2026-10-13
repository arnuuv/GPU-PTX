"""Greedy register allocation from a liveness interference graph.

One virtual register keeps one physical name for the whole kernel. Two virtual
registers share a physical register only when no program point has both live.
PTX allows a destination to alias one of its sources, so a virtual register
does not interfere with itself.
"""

from __future__ import annotations

from ptxlab.dtype import DType
from ptxlab.ir import Cbr, Instr, KernelIR, VReg


def greedy_map(kernel: KernelIR) -> dict[str, str]:
    """Map every virtual register name to a physical PTX name."""

    dtypes = _dtypes(kernel)
    if not dtypes:
        return {}
    graph = _interference(kernel)
    order = sorted(dtypes, key=lambda name: (-len(graph.get(name, ())), name))
    color: dict[str, int] = {}
    for name in order:
        used = {color[other] for other in graph.get(name, ()) if other in color and dtypes[other] is dtypes[name]}
        chosen = 0
        while chosen in used:
            chosen += 1
        color[name] = chosen
    return {name: f"{dtypes[name].ptx_reg_prefix}{index}" for name, index in color.items()}


def physical_counts(mapping: dict[str, str], kernel: KernelIR) -> dict[DType, int]:
    dtypes = _dtypes(kernel)
    counts = {dtype: 0 for dtype in DType}
    for name, physical in mapping.items():
        dtype = dtypes[name]
        index = int(physical.removeprefix(dtype.ptx_reg_prefix))
        counts[dtype] = max(counts[dtype], index + 1)
    return counts


def _dtypes(kernel: KernelIR) -> dict[str, DType]:
    found: dict[str, DType] = {}
    for block in kernel.blocks:
        for instr in block.instrs:
            for reg in _mentioned(instr):
                found[reg.name] = reg.dtype
        if isinstance(block.term, Cbr):
            found[block.term.pred.name] = block.term.pred.dtype
    return found


def _interference(kernel: KernelIR) -> dict[str, set[str]]:
    graph: dict[str, set[str]] = {}

    def link(left: str, right: str) -> None:
        if left == right:
            return
        graph.setdefault(left, set()).add(right)
        graph.setdefault(right, set()).add(left)

    def clique(names: set[str]) -> None:
        ordered = sorted(names)
        for index, left in enumerate(ordered):
            graph.setdefault(left, set())
            for right in ordered[index + 1 :]:
                link(left, right)

    by_label = {block.label: block for block in kernel.blocks}
    succs = {block.label: _succs(block) for block in kernel.blocks}
    use, defs = {}, {}
    for block in kernel.blocks:
        used: set[str] = set()
        defined: set[str] = set()
        for instr in block.instrs:
            for reg in _uses(instr):
                if reg.name not in defined:
                    used.add(reg.name)
            if instr.dst is not None:
                defined.add(instr.dst.name)
        if isinstance(block.term, Cbr) and block.term.pred.name not in defined:
            used.add(block.term.pred.name)
        use[block.label] = used
        defs[block.label] = defined
    live_in = {block.label: set() for block in kernel.blocks}
    changed = True
    while changed:
        changed = False
        for block in reversed(kernel.blocks):
            out: set[str] = set()
            for succ in succs[block.label]:
                out |= live_in[succ]
            inn = use[block.label] | (out - defs[block.label])
            if inn != live_in[block.label]:
                live_in[block.label] = inn
                changed = True
    for block in kernel.blocks:
        out = set()
        for succ in succs[block.label]:
            out |= live_in[succ]
        live = set(out)
        if isinstance(block.term, Cbr):
            live.add(block.term.pred.name)
        clique(live)
        for instr in reversed(block.instrs):
            uses = {reg.name for reg in _uses(instr)}
            if instr.dst is not None:
                for other in uses | live:
                    link(instr.dst.name, other)
                live.discard(instr.dst.name)
            live |= uses
            clique(live)
        _ = by_label
    return graph


def _succs(block) -> list[str]:
    term = block.term
    if isinstance(term, Cbr):
        return [term.then_target, term.else_target]
    target = getattr(term, "target", None)
    return [target] if isinstance(target, str) else []


def _uses(instr: Instr) -> list[VReg]:
    return [arg for arg in instr.args if isinstance(arg, VReg)]


def _mentioned(instr: Instr) -> list[VReg]:
    found = _uses(instr)
    if instr.dst is not None:
        found.append(instr.dst)
    return found
