"""Parser for the PTX subset emitted by :mod:`ptxlab.emit`.

This is not a general PTX front end. Directives, types, and opcodes outside
the tables below raise ``ParseError`` or ``UnsupportedInstruction``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np

from ptxlab.errors import ParseError, UnsupportedInstruction
from ptxlab.toolchain import parse_version, validate_toolchain

_REG = re.compile(r"%[A-Za-z_][A-Za-z0-9_]*(?:\.[xyz])?$")
_IDENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]*$")
_HEX_F32 = re.compile(r"0f[0-9a-fA-F]{8}$")
_INT = re.compile(r"-?\d+$")

_PREFIXES = ("%rd", "%rs", "%r", "%f", "%p")

ALLOWED_OPS = {
    "mov.u32",
    "mov.u64",
    "mov.s32",
    "mov.f32",
    "mov.pred",
    "ld.param.u32",
    "ld.param.u64",
    "ld.param.s32",
    "ld.param.f32",
    "mul.lo.u32",
    "mul.lo.s32",
    "mul.rn.f32",
    "mul.wide.u32",
    "add.u32",
    "add.u64",
    "add.s32",
    "add.rn.f32",
    "sub.u32",
    "sub.s32",
    "sub.rn.f32",
    "mad.lo.u32",
    "mad.lo.s32",
    "shr.u32",
    "and.pred",
    "or.pred",
    "not.pred",
    "bar.sync",
    "bra",
    "ret",
}
ALLOWED_OPS.update(
    f"setp.{cmp}.{ty}" for cmp in ("eq", "ne", "lt", "le", "gt", "ge") for ty in ("u32", "s32")
)
ALLOWED_OPS.update(
    f"{op}.{ty}"
    for op in ("ld.global", "st.global", "ld.shared", "st.shared")
    for ty in ("f32", "s32", "u32")
)


@dataclass(frozen=True)
class PParam:
    name: str
    kind: str  # u64, u32, s32, f32


@dataclass(frozen=True)
class PShared:
    name: str
    kind: str
    length: int

    @property
    def nbytes(self) -> int:
        return self.length * 4


@dataclass(frozen=True)
class SInst:
    op: str
    dst: str | None
    srcs: tuple[object, ...]
    pred: str | None = None
    pred_not: bool = False
    line: int = 0


@dataclass
class Program:
    version: tuple[int, int]
    target: str
    address_size: int
    name: str
    params: list[PParam]
    shared: list[PShared]
    reg_counts: dict[str, int]
    expect_block: tuple[int, int, int] | None
    insts: list[SInst] = field(default_factory=list)
    labels: dict[str, int] = field(default_factory=dict)

    def param_map(self) -> dict[str, PParam]:
        return {param.name: param for param in self.params}


def parse(text: str) -> Program:
    version: tuple[int, int] | None = None
    target: str | None = None
    address_size: int | None = None
    name: str | None = None
    params: list[PParam] = []
    shared: list[PShared] = []
    reg_counts: dict[str, int] = {}
    expect_block: tuple[int, int, int] | None = None
    insts: list[SInst] = []
    labels: dict[str, int] = {}
    pending: list[str] = []
    state = "top"

    for lineno, raw in enumerate(text.splitlines(), start=1):
        code, comment = _split_comment(raw)
        block = _block_comment(comment)
        if block is not None:
            expect_block = block
        line = code.strip().rstrip(";").strip()
        if not line:
            continue
        if line.endswith(";") and code.strip().endswith(";"):
            line = code.strip()[:-1].strip()
        try:
            if state == "top":
                if line.startswith(".version "):
                    version = parse_version(line.split()[1])
                elif line.startswith(".target "):
                    target = line.split()[1]
                elif line.startswith(".address_size "):
                    address_size = int(line.split()[1])
                elif line.startswith(".visible .entry "):
                    name = line.removeprefix(".visible .entry ").replace("(", "").strip()
                    state = "params"
                else:
                    raise ParseError(f"line {lineno}: unexpected directive {line!r}")
                continue
            if state == "params":
                if line.startswith(".param "):
                    parts = line.replace(",", "").split()
                    if len(parts) != 3 or not parts[1].startswith("."):
                        raise ParseError(f"line {lineno}: bad parameter {line!r}")
                    params.append(PParam(parts[2], parts[1][1:]))
                elif line == ")":
                    state = "body-open"
                else:
                    raise ParseError(f"line {lineno}: expected a parameter, got {line!r}")
                continue
            if state == "body-open":
                if line != "{":
                    raise ParseError(f"line {lineno}: expected '{{'")
                state = "body"
                continue
            if line == "}":
                state = "done"
                continue
            if state == "done":
                raise ParseError(f"line {lineno}: trailing text after kernel")
            if line.startswith(".reg "):
                prefix, count = _parse_reg(line, lineno)
                reg_counts[prefix] = count
                continue
            if line.startswith(".shared "):
                shared.append(_parse_shared(line, lineno))
                continue
            if line.endswith(":"):
                label = line[:-1]
                if not _IDENT.match(label):
                    raise ParseError(f"line {lineno}: bad label {label!r}")
                if label in labels:
                    raise ParseError(f"line {lineno}: duplicate label {label}")
                pending.append(label)
                continue
            inst = _parse_inst(line, lineno)
            while pending:
                labels[pending.pop(0)] = len(insts)
            insts.append(inst)
        except (UnsupportedInstruction, ParseError):
            raise
        except Exception as exc:
            raise ParseError(f"line {lineno}: {exc}") from exc

    if state != "done" or version is None or target is None or address_size is None or name is None:
        raise ParseError("incomplete PTX kernel")
    validate_toolchain(version, target, address_size)
    if pending:
        raise ParseError(f"label {pending[0]} is not attached to an instruction")
    program = Program(
        version=version,
        target=target,
        address_size=address_size,
        name=name,
        params=params,
        shared=shared,
        reg_counts=reg_counts,
        expect_block=expect_block,
        insts=insts,
        labels=labels,
    )
    _check_registers(program)
    _check_labels(program)
    return program


def _split_comment(raw: str) -> tuple[str, str]:
    if "//" not in raw:
        return raw, ""
    code, comment = raw.split("//", 1)
    return code, comment.strip()


def _block_comment(comment: str) -> tuple[int, int, int] | None:
    if not comment.startswith(".ptxlab_block"):
        return None
    parts = comment.split()
    if len(parts) != 4:
        raise ParseError(f"bad .ptxlab_block comment {comment!r}")
    return (int(parts[1]), int(parts[2]), int(parts[3]))


def _parse_reg(line: str, lineno: int) -> tuple[str, int]:
    match = re.fullmatch(r"\.reg\s+\.(?:pred|u32|u64|s32|f32)\s+(%[A-Za-z]+)<(\d+)>", line)
    if match is None:
        raise ParseError(f"line {lineno}: bad register declaration {line!r}")
    prefix, count = match.group(1), int(match.group(2))
    if count <= 0:
        raise ParseError(f"line {lineno}: empty register declaration")
    return prefix, count


def _parse_shared(line: str, lineno: int) -> PShared:
    match = re.fullmatch(r"\.shared\s+\.align\s+4\s+\.(f32|s32|u32)\s+([A-Za-z_]\w*)\[(\d+)\]", line)
    if match is None:
        raise ParseError(f"line {lineno}: bad shared declaration {line!r}")
    return PShared(match.group(2), match.group(1), int(match.group(3)))


def _parse_inst(line: str, lineno: int) -> SInst:
    pred: str | None = None
    pred_not = False
    if line.startswith("@"):
        match = re.match(r"@(!)?(%p\d+)\s+(.*)$", line)
        if match is None:
            raise ParseError(f"line {lineno}: bad predicate {line!r}")
        pred_not = match.group(1) == "!"
        pred = match.group(2)
        line = match.group(3)
    if line == "ret":
        return SInst("ret", None, (), pred, pred_not, lineno)
    if line.startswith("bra "):
        target = line[4:].strip()
        if not _IDENT.match(target):
            raise ParseError(f"line {lineno}: bad branch target {target!r}")
        return SInst("bra", None, (target,), pred, pred_not, lineno)
    if line.startswith("bar.sync "):
        return SInst("bar.sync", None, (int(line.split()[1]),), pred, pred_not, lineno)
    parts = line.split(None, 1)
    if len(parts) != 2:
        raise UnsupportedInstruction(f"line {lineno}: unsupported instruction {line!r}")
    op, rest = parts
    if op not in ALLOWED_OPS:
        raise UnsupportedInstruction(f"line {lineno}: unsupported instruction {op}")
    operands = [item.strip() for item in rest.split(",")]
    if op.startswith("st."):
        if len(operands) != 2 or not operands[0].startswith("["):
            raise ParseError(f"line {lineno}: bad store {line!r}")
        return SInst(op, None, (_mem(operands[0], lineno), _atom(operands[1], lineno)), pred, pred_not, lineno)
    dst = _atom(operands[0], lineno)
    if not isinstance(dst, str) or not dst.startswith("%"):
        raise ParseError(f"line {lineno}: instruction needs a destination register")
    srcs = tuple(_mem(item, lineno) if item.startswith("[") else _atom(item, lineno) for item in operands[1:])
    return SInst(op, dst, srcs, pred, pred_not, lineno)


def _mem(token: str, lineno: int) -> str:
    if not (token.startswith("[") and token.endswith("]")):
        raise ParseError(f"line {lineno}: expected a bracketed operand, got {token!r}")
    inner = _atom(token[1:-1].strip(), lineno)
    if not isinstance(inner, str):
        raise ParseError(f"line {lineno}: memory operand must name a register or symbol")
    return inner


def _atom(token: str, lineno: int) -> object:
    if _REG.match(token):
        return token
    if _HEX_F32.match(token):
        bits = np.uint32(int(token[2:], 16))
        return float(np.array(bits).view(np.float32))
    if _INT.match(token):
        return int(token)
    if _IDENT.match(token):
        return token
    raise ParseError(f"line {lineno}: bad operand {token!r}")


def _check_registers(program: Program) -> None:
    special = re.compile(r"%(?:tid|ntid|ctaid)\.[xyz]$")
    declared = set(program.reg_counts)
    for inst in program.insts:
        for reg in _regs(inst):
            if special.match(reg):
                continue
            prefix = _prefix(reg)
            if prefix is None or prefix not in declared:
                raise ParseError(f"line {inst.line}: register {reg} is not declared")
            number = int(reg[len(prefix) :])
            if number >= program.reg_counts[prefix]:
                raise ParseError(f"line {inst.line}: register {reg} is outside its declaration")


def _check_labels(program: Program) -> None:
    for inst in program.insts:
        if inst.op == "bra":
            target = inst.srcs[0]
            if target not in program.labels:
                raise ParseError(f"line {inst.line}: unknown label {target}")


def _regs(inst: SInst) -> list[str]:
    found: list[str] = []
    if inst.pred:
        found.append(inst.pred)
    if isinstance(inst.dst, str) and inst.dst.startswith("%"):
        found.append(inst.dst)
    for src in inst.srcs:
        if isinstance(src, str) and src.startswith("%"):
            found.append(src)
    return found


def _prefix(reg: str) -> str | None:
    for prefix in _PREFIXES:
        if reg.startswith(prefix) and reg[len(prefix) :].isdigit():
            return prefix
    return None
