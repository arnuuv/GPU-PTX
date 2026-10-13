"""Structured kernel AST and a small expression builder."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Iterator

import numpy as np

from ptxlab.dtype import DType
from ptxlab.errors import LoweringError
from ptxlab.ir import Dim
from ptxlab.toolchain import DEFAULT_ADDRESS_SIZE, DEFAULT_ISA, DEFAULT_TARGET

_IDENT = __import__("re").compile(r"[A-Za-z_][A-Za-z0-9_]*$")


@dataclass(frozen=True)
class Expr:
    dtype: DType

    def __bool__(self) -> bool:
        raise TypeError("use when() for PTXLab predicates; expressions are not Python booleans")

    def __add__(self, other: Expr | int | float) -> Expr:
        return _arith("add", self, other)

    def __radd__(self, other: Expr | int | float) -> Expr:
        return _arith("add", other, self)

    def __sub__(self, other: Expr | int | float) -> Expr:
        return _arith("sub", self, other)

    def __rsub__(self, other: Expr | int | float) -> Expr:
        return _arith("sub", other, self)

    def __mul__(self, other: Expr | int | float) -> Expr:
        return _arith("mul", self, other)

    def __rmul__(self, other: Expr | int | float) -> Expr:
        return _arith("mul", other, self)

    def __lt__(self, other: Expr | int | float) -> Expr:
        return _cmp("lt", self, other)

    def __le__(self, other: Expr | int | float) -> Expr:
        return _cmp("le", self, other)

    def __gt__(self, other: Expr | int | float) -> Expr:
        return _cmp("gt", self, other)

    def __ge__(self, other: Expr | int | float) -> Expr:
        return _cmp("ge", self, other)

    def __eq__(self, other: object) -> Expr:  # type: ignore[override]
        if not isinstance(other, (Expr, int, float)) or isinstance(other, bool):
            return NotImplemented
        return _cmp("eq", self, other)

    def __ne__(self, other: object) -> Expr:  # type: ignore[override]
        if not isinstance(other, (Expr, int, float)) or isinstance(other, bool):
            return NotImplemented
        return _cmp("ne", self, other)


@dataclass(frozen=True, eq=False)
class ConstExpr(Expr):
    value: int | float


@dataclass(frozen=True, eq=False)
class VarExpr(Expr):
    name: str


@dataclass(frozen=True, eq=False)
class ScalarExpr(Expr):
    name: str


@dataclass(frozen=True, eq=False)
class BinExpr(Expr):
    op: str
    left: Expr
    right: Expr


@dataclass(frozen=True, eq=False)
class CmpExpr(Expr):
    op: str
    left: Expr
    right: Expr
    operand_dtype: DType


@dataclass(frozen=True, eq=False)
class PredExpr(Expr):
    op: str
    args: tuple[Expr, ...]


@dataclass(frozen=True, eq=False)
class IndexExpr(Expr):
    kind: str
    dim: str


@dataclass(frozen=True, eq=False)
class LoadExpr(Expr):
    array: str
    indices: tuple[Expr, ...]


@dataclass(frozen=True, eq=False)
class LoadSharedExpr(Expr):
    buffer: str
    indices: tuple[Expr, ...]


@dataclass(frozen=True, eq=False)
class MadExpr(Expr):
    a: Expr
    b: Expr
    c: Expr


@dataclass(frozen=True, eq=False)
class ShrExpr(Expr):
    value: Expr
    amount: Expr


@dataclass(frozen=True, eq=False)
class FmaExpr(Expr):
    a: Expr
    b: Expr
    c: Expr


@dataclass(frozen=True, eq=False)
class UnaryExpr(Expr):
    op: str
    value: Expr


@dataclass(frozen=True, eq=False)
class AtomExpr(Expr):
    array: str
    indices: tuple[Expr, ...]
    value: Expr


@dataclass(frozen=True)
class ArrayRef:
    name: str
    dtype: DType
    shape: tuple[Dim, ...]


@dataclass(frozen=True)
class SharedRef:
    name: str
    dtype: DType
    shape: tuple[int, ...]


@dataclass
class AssignStmt:
    name: str
    dtype: DType
    expr: Expr


@dataclass
class StoreStmt:
    array: str
    indices: tuple[Expr, ...]
    value: Expr


@dataclass
class StoreSharedStmt:
    buffer: str
    indices: tuple[Expr, ...]
    value: Expr


@dataclass
class IfStmt:
    cond: Expr
    then_body: list[Stmt] = field(default_factory=list)
    else_body: list[Stmt] = field(default_factory=list)


@dataclass
class ForStmt:
    var: str
    start: Expr
    end: Expr
    step: Expr
    body: list[Stmt] = field(default_factory=list)


@dataclass
class BarrierStmt:
    bar_id: int = 0


@dataclass
class LoadV4Stmt:
    array: str
    index: Expr
    names: tuple[str, str, str, str]


@dataclass
class StoreV4Stmt:
    array: str
    index: Expr
    values: tuple[Expr, Expr, Expr, Expr]


Stmt = AssignStmt | StoreStmt | StoreSharedStmt | IfStmt | ForStmt | BarrierStmt | LoadV4Stmt | StoreV4Stmt


@dataclass
class ParamDecl:
    name: str
    dtype: DType
    shape: tuple[Dim, ...] = ()


@dataclass
class SharedDeclAST:
    name: str
    dtype: DType
    shape: tuple[int, ...]


@dataclass
class Kernel:
    name: str
    params: list[ParamDecl]
    shared: list[SharedDeclAST]
    body: list[Stmt]
    version: tuple[int, int] = DEFAULT_ISA
    target: str = DEFAULT_TARGET
    address_size: int = DEFAULT_ADDRESS_SIZE
    expect_block: tuple[int, int, int] | None = None


class _Otherwise:
    def __init__(self, builder: KernelBuilder, stmt: IfStmt) -> None:
        self._builder = builder
        self._stmt = stmt

    @contextmanager
    def otherwise(self) -> Iterator[None]:
        self._builder._stack.append(self._stmt.else_body)
        try:
            yield
        finally:
            self._builder._stack.pop()


class KernelBuilder:
    """Builds a Kernel AST. Control flow is structured: when() and for_range()."""

    def __init__(
        self,
        name: str,
        *,
        version: tuple[int, int] = DEFAULT_ISA,
        target: str = DEFAULT_TARGET,
        address_size: int = DEFAULT_ADDRESS_SIZE,
        expect_block: tuple[int, int, int] | None = None,
    ) -> None:
        _check_name(name)
        self.name = name
        self.version = version
        self.target = target
        self.address_size = address_size
        self.expect_block = expect_block
        self._params: list[ParamDecl] = []
        self._shared: list[SharedDeclAST] = []
        self._arrays: dict[str, ArrayRef] = {}
        self._shared_map: dict[str, SharedRef] = {}
        self._scalars: dict[str, ScalarExpr] = {}
        self._body: list[Stmt] = []
        self._stack: list[list[Stmt]] = [self._body]
        self._locals: dict[str, DType] = {}
        self._seq = 0

    def array(self, name: str, dtype: DType, shape: tuple[str | int, ...]) -> ArrayRef:
        _check_name(name)
        if dtype not in (DType.F32, DType.F16, DType.S32, DType.U32):
            raise LoweringError("array elements must be f32, f16, s32, or u32")
        if not shape:
            raise LoweringError("array parameters need a shape; use scalar() for rank 0")
        dims = tuple(_dim(item) for item in shape)
        ref = ArrayRef(name, dtype, dims)
        self._params.append(ParamDecl(name, dtype, dims))
        self._arrays[name] = ref
        return ref

    def scalar(self, name: str, dtype: DType) -> ScalarExpr:
        _check_name(name)
        if dtype not in (DType.F32, DType.S32, DType.U32):
            raise LoweringError("scalars must be f32, s32, or u32")
        expr = ScalarExpr(dtype, name)
        self._params.append(ParamDecl(name, dtype))
        self._scalars[name] = expr
        return expr

    def shared(self, name: str, dtype: DType, shape: tuple[int, ...] | int) -> SharedRef:
        _check_name(name)
        if dtype not in (DType.F32, DType.S32, DType.U32):
            raise LoweringError("shared elements must be f32, s32, or u32")
        dims = (shape,) if isinstance(shape, int) else tuple(shape)
        if not dims or any(dim <= 0 for dim in dims):
            raise LoweringError("shared shape must be positive and static")
        ref = SharedRef(name, dtype, dims)
        self._shared.append(SharedDeclAST(name, dtype, dims))
        self._shared_map[name] = ref
        return ref

    def f32(self, value: int | float) -> ConstExpr:
        return ConstExpr(DType.F32, float(np.float32(value)))

    def s32(self, value: int) -> ConstExpr:
        _check_int(value, DType.S32)
        return ConstExpr(DType.S32, value)

    def u32(self, value: int) -> ConstExpr:
        _check_int(value, DType.U32)
        return ConstExpr(DType.U32, value)

    def tid(self, dim: str = "x") -> IndexExpr:
        return IndexExpr(DType.U32, "tid", _axis(dim))

    def ctaid(self, dim: str = "x") -> IndexExpr:
        return IndexExpr(DType.U32, "ctaid", _axis(dim))

    def ntid(self, dim: str = "x") -> IndexExpr:
        return IndexExpr(DType.U32, "ntid", _axis(dim))

    def gid(self, dim: str = "x") -> Expr:
        return self.ctaid(dim) * self.ntid(dim) + self.tid(dim)

    def let(self, name: str, expr: Expr) -> VarExpr:
        _check_name(name)
        if name in self._locals and self._locals[name] is not expr.dtype:
            raise LoweringError(f"{name} is already {self._locals[name].value}")
        self._locals[name] = expr.dtype
        self._emit(AssignStmt(name, expr.dtype, expr))
        return VarExpr(expr.dtype, name)

    def assign(self, var: VarExpr, expr: Expr) -> None:
        if var.dtype is not expr.dtype:
            raise LoweringError(f"cannot assign {expr.dtype.value} to {var.name}:{var.dtype.value}")
        self._emit(AssignStmt(var.name, var.dtype, expr))

    def load(self, array: ArrayRef, indices: tuple[Expr | int, ...]) -> LoadExpr:
        idxs = _indices(array.name, array.shape, indices)
        return LoadExpr(array.dtype, array.name, idxs)

    def store(self, array: ArrayRef, indices: tuple[Expr | int, ...], value: Expr) -> None:
        if value.dtype is not array.dtype:
            raise LoweringError(f"store to {array.name} expects {array.dtype.value}")
        self._emit(StoreStmt(array.name, _indices(array.name, array.shape, indices), value))

    def load_shared(self, buf: SharedRef, indices: tuple[Expr | int, ...]) -> LoadSharedExpr:
        return LoadSharedExpr(buf.dtype, buf.name, _indices(buf.name, buf.shape, indices))

    def store_shared(self, buf: SharedRef, indices: tuple[Expr | int, ...], value: Expr) -> None:
        if value.dtype is not buf.dtype:
            raise LoweringError(f"shared store to {buf.name} expects {buf.dtype.value}")
        self._emit(StoreSharedStmt(buf.name, _indices(buf.name, buf.shape, indices), value))

    def mad_lo(self, a: Expr | int, b: Expr | int, c: Expr | int) -> MadExpr:
        exprs = [_require(item) if isinstance(item, Expr) else None for item in (a, b, c)]
        dtype = next((item.dtype for item in exprs if item is not None), None)
        if dtype not in (DType.S32, DType.U32):
            raise LoweringError("mad.lo supports s32 and u32")
        args = tuple(_coerce(item, dtype) for item in (a, b, c))
        return MadExpr(dtype, *args)

    def fma(self, a: Expr | int | float, b: Expr | int | float, c: Expr | int | float) -> FmaExpr:
        return FmaExpr(DType.F32, _coerce(a, DType.F32), _coerce(b, DType.F32), _coerce(c, DType.F32))

    def div(self, left: Expr | int | float, right: Expr | int | float) -> BinExpr:
        return BinExpr(DType.F32, "div", _coerce(left, DType.F32), _coerce(right, DType.F32))

    def sqrt(self, value: Expr | int | float) -> UnaryExpr:
        return UnaryExpr(DType.F32, "sqrt", _coerce(value, DType.F32))

    def maximum(self, left: Expr | int | float, right: Expr | int | float) -> BinExpr:
        return BinExpr(DType.F32, "max", _coerce(left, DType.F32), _coerce(right, DType.F32))

    def ex2(self, value: Expr | int | float) -> UnaryExpr:
        return UnaryExpr(DType.F32, "ex2", _coerce(value, DType.F32))

    def to_f32(self, value: Expr) -> UnaryExpr:
        if value.dtype is not DType.F16:
            raise LoweringError("to_f32 expects f16")
        return UnaryExpr(DType.F32, "cvt_f32", value)

    def to_f16(self, value: Expr) -> UnaryExpr:
        if value.dtype is not DType.F32:
            raise LoweringError("to_f16 expects f32")
        return UnaryExpr(DType.F16, "cvt_f16", value)

    def atomic_add(self, array: ArrayRef, indices: tuple[Expr | int, ...], value: Expr) -> AtomExpr:
        if array.dtype not in (DType.U32, DType.F32) or value.dtype is not array.dtype:
            raise LoweringError("atomic_add supports u32 and f32 arrays")
        return AtomExpr(array.dtype, array.name, _indices(array.name, array.shape, indices), value)

    def load_v4(self, array: ArrayRef, index: Expr | int) -> tuple[VarExpr, VarExpr, VarExpr, VarExpr]:
        if array.dtype is not DType.F32 or len(array.shape) != 1:
            raise LoweringError("load_v4 supports 1D f32 arrays")
        names = tuple(self._fresh(DType.F32) for _ in range(4))
        self._emit(LoadV4Stmt(array.name, _coerce(index, DType.U32), names))  # type: ignore[arg-type]
        return tuple(VarExpr(DType.F32, name) for name in names)  # type: ignore[return-value]

    def store_v4(self, array: ArrayRef, index: Expr | int, values: tuple[Expr, Expr, Expr, Expr]) -> None:
        if array.dtype is not DType.F32 or any(value.dtype is not DType.F32 for value in values):
            raise LoweringError("store_v4 expects four f32 values")
        self._emit(StoreV4Stmt(array.name, _coerce(index, DType.U32), values))

    def shr(self, value: Expr | int, amount: Expr | int) -> ShrExpr:
        return ShrExpr(DType.U32, _coerce(value, DType.U32), _coerce(amount, DType.U32))

    def land(self, left: Expr, right: Expr) -> PredExpr:
        return _pred2("and", left, right)

    def lor(self, left: Expr, right: Expr) -> PredExpr:
        return _pred2("or", left, right)

    def lnot(self, value: Expr) -> PredExpr:
        if value.dtype is not DType.PRED:
            raise LoweringError("not expects a predicate")
        return PredExpr(DType.PRED, "not", (value,))

    def barrier(self, bar_id: int = 0) -> None:
        if not isinstance(bar_id, int) or not 0 <= bar_id <= 15:
            raise LoweringError("barrier id must be in 0..15")
        self._emit(BarrierStmt(bar_id))

    @contextmanager
    def when(self, cond: Expr) -> Iterator[_Otherwise]:
        if cond.dtype is not DType.PRED:
            raise LoweringError("when() expects a predicate")
        stmt = IfStmt(cond, [], [])
        self._emit(stmt)
        self._stack.append(stmt.then_body)
        try:
            yield _Otherwise(self, stmt)
        finally:
            self._stack.pop()

    @contextmanager
    def for_range(
        self,
        name: str,
        start: Expr | int,
        end: Expr | int,
        step: Expr | int = 1,
    ) -> Iterator[VarExpr]:
        _check_name(name)
        if name in self._locals and self._locals[name] is not DType.U32:
            raise LoweringError(f"{name} is already {self._locals[name].value}")
        self._locals[name] = DType.U32
        stmt = ForStmt(name, _coerce(start, DType.U32), _coerce(end, DType.U32), _coerce(step, DType.U32), [])
        self._emit(stmt)
        self._stack.append(stmt.body)
        try:
            yield VarExpr(DType.U32, name)
        finally:
            self._stack.pop()

    def build(self) -> Kernel:
        if len(self._stack) != 1:
            raise LoweringError("cannot build a kernel inside when() or for_range()")
        return Kernel(
            name=self.name,
            params=list(self._params),
            shared=list(self._shared),
            body=list(self._body),
            version=self.version,
            target=self.target,
            address_size=self.address_size,
            expect_block=self.expect_block,
        )

    def _emit(self, stmt: Stmt) -> None:
        self._stack[-1].append(stmt)

    def _fresh(self, dtype: DType) -> str:
        self._seq += 1
        name = f"vec{self._seq}"
        self._locals[name] = dtype
        return name


def _arith(op: str, left: Expr | int | float, right: Expr | int | float) -> BinExpr:
    lhs = left if isinstance(left, Expr) else None
    rhs = right if isinstance(right, Expr) else None
    if lhs is None and rhs is None:
        raise LoweringError("arithmetic needs a typed operand")
    dtype = lhs.dtype if lhs is not None else rhs.dtype  # type: ignore[union-attr]
    if dtype not in (DType.F32, DType.S32, DType.U32):
        raise LoweringError(f"{op} supports f32, s32, and u32 in the DSL")
    return BinExpr(dtype, op, _coerce(left, dtype), _coerce(right, dtype))


def _cmp(op: str, left: Expr | int | float, right: Expr | int | float) -> CmpExpr:
    lhs = left if isinstance(left, Expr) else None
    rhs = right if isinstance(right, Expr) else None
    if lhs is None and rhs is None:
        raise LoweringError("comparison needs a typed operand")
    dtype = lhs.dtype if lhs is not None else rhs.dtype  # type: ignore[union-attr]
    if dtype not in (DType.S32, DType.U32):
        raise LoweringError("comparisons support s32 and u32")
    return CmpExpr(DType.PRED, op, _coerce(left, dtype), _coerce(right, dtype), dtype)


def _pred2(op: str, left: Expr, right: Expr) -> PredExpr:
    if left.dtype is not DType.PRED or right.dtype is not DType.PRED:
        raise LoweringError(f"{op} expects predicates")
    return PredExpr(DType.PRED, op, (left, right))


def _require(expr: Expr) -> Expr:
    return expr


def _coerce(value: Expr | int | float, dtype: DType) -> Expr:
    if isinstance(value, Expr):
        if value.dtype is not dtype:
            raise LoweringError(f"expected {dtype.value}, got {value.dtype.value}")
        return value
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise LoweringError(f"cannot use {value!r} as {dtype.value}")
    if dtype is DType.F32:
        return ConstExpr(DType.F32, float(np.float32(value)))
    if isinstance(value, float):
        raise LoweringError(f"{dtype.value} constants must be integers")
    _check_int(value, dtype)
    return ConstExpr(dtype, value)


def _check_int(value: int, dtype: DType) -> None:
    if dtype is DType.U32 and not 0 <= value < 2**32:
        raise LoweringError(f"{value} does not fit in u32")
    if dtype is DType.S32 and not -(2**31) <= value < 2**31:
        raise LoweringError(f"{value} does not fit in s32")


def _dim(item: str | int) -> Dim:
    if isinstance(item, int):
        if item < 0:
            raise LoweringError("static shape extents must be non-negative")
        return Dim(value=item)
    if isinstance(item, str):
        _check_name(item)
        return Dim(name=item)
    raise LoweringError("shape extents must be parameter names or integers")


def _axis(dim: str) -> str:
    if dim not in ("x", "y", "z"):
        raise LoweringError("thread axes are x, y, and z")
    return dim


def _check_name(name: str) -> None:
    if not _IDENT.match(name):
        raise LoweringError(f"invalid name {name!r}")


def _indices(name: str, shape: tuple[Dim, ...] | tuple[int, ...], indices: tuple[Expr | int, ...]) -> tuple[Expr, ...]:
    if len(indices) != len(shape):
        raise LoweringError(f"{name} expects {len(shape)} indices, got {len(indices)}")
    return tuple(_coerce(index, DType.U32) for index in indices)
