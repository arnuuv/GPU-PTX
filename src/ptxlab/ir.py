"""Typed three-address IR lowered to the supported PTX subset."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from ptxlab.dtype import DType
from ptxlab.toolchain import DEFAULT_ADDRESS_SIZE, DEFAULT_ISA, DEFAULT_TARGET


class Op(Enum):
    CONST = "const"
    MOV = "mov"
    ADD = "add"
    SUB = "sub"
    MUL = "mul"
    MUL_WIDE = "mul_wide"
    MAD_LO = "mad_lo"
    FMA = "fma"
    DIV = "div"
    SQRT = "sqrt"
    MAX = "max"
    EX2 = "ex2"
    CVT = "cvt"
    SHR = "shr"
    CMP = "cmp"
    AND_PRED = "and_pred"
    OR_PRED = "or_pred"
    NOT_PRED = "not_pred"
    TID = "tid"
    CTAID = "ctaid"
    NTID = "ntid"
    LD_PARAM = "ld_param"
    LD_GLOBAL = "ld_global"
    ST_GLOBAL = "st_global"
    LD_GLOBAL_V4 = "ld_global_v4"
    ST_GLOBAL_V4 = "st_global_v4"
    ATOM_ADD = "atom_add"
    SHARED_BASE = "shared_base"
    LD_SHARED = "ld_shared"
    ST_SHARED = "st_shared"
    BAR = "bar"


CMP_CODES = ("eq", "ne", "lt", "le", "gt", "ge")
DIMS = ("x", "y", "z")


@dataclass(frozen=True)
class VReg:
    name: str
    dtype: DType

    def __post_init__(self) -> None:
        if not self.name.startswith("%"):
            raise ValueError(f"virtual register {self.name!r} must start with '%'")


@dataclass(frozen=True)
class Dim:
    """A tensor extent. Exactly one of ``name`` or ``value`` is set."""

    name: str | None = None
    value: int | None = None

    def __post_init__(self) -> None:
        if (self.name is None) == (self.value is None):
            raise ValueError("Dim needs exactly one of name or value")
        if self.value is not None and self.value < 0:
            raise ValueError("static dimensions must be non-negative")

    def __str__(self) -> str:
        return self.name if self.name is not None else str(self.value)


@dataclass(frozen=True)
class Param:
    name: str
    dtype: DType
    shape: tuple[Dim, ...] = ()

    @property
    def is_pointer(self) -> bool:
        return bool(self.shape)


@dataclass(frozen=True)
class SharedDecl:
    name: str
    dtype: DType
    shape: tuple[int, ...]

    @property
    def length(self) -> int:
        size = 1
        for dim in self.shape:
            size *= dim
        return size

    @property
    def nbytes(self) -> int:
        return self.length * self.dtype.size_bytes


@dataclass(frozen=True)
class Instr:
    op: Op
    dst: VReg | None
    args: tuple[object, ...] = ()
    dtype: DType | None = None
    cmp: str | None = None
    dim: str | None = None
    symbol: str | None = None
    extra_dsts: tuple[VReg, ...] = ()


@dataclass
class Block:
    label: str
    instrs: list[Instr] = field(default_factory=list)
    term: Term | None = None


@dataclass(frozen=True)
class Br:
    target: str


@dataclass(frozen=True)
class Cbr:
    pred: VReg
    then_target: str
    else_target: str


@dataclass(frozen=True)
class Ret:
    pass


Term = Br | Cbr | Ret


@dataclass
class KernelIR:
    name: str
    params: list[Param]
    shared: list[SharedDecl]
    blocks: list[Block]
    version: tuple[int, int] = DEFAULT_ISA
    target: str = DEFAULT_TARGET
    address_size: int = DEFAULT_ADDRESS_SIZE
    expect_block: tuple[int, int, int] | None = None

    def param_map(self) -> dict[str, Param]:
        return {param.name: param for param in self.params}

    def shared_map(self) -> dict[str, SharedDecl]:
        return {item.name: item for item in self.shared}
