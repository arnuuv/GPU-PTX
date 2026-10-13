"""Lower the kernel AST into typed three-address IR."""

from __future__ import annotations

from ptxlab.dsl import (
    ArrayRef,
    AssignStmt,
    BarrierStmt,
    BinExpr,
    CmpExpr,
    ConstExpr,
    Expr,
    ForStmt,
    IfStmt,
    IndexExpr,
    Kernel,
    LoadExpr,
    LoadSharedExpr,
    MadExpr,
    PredExpr,
    ScalarExpr,
    SharedRef,
    ShrExpr,
    StoreSharedStmt,
    StoreStmt,
    VarExpr,
)
from ptxlab.dtype import DType
from ptxlab.errors import LoweringError
from ptxlab.ir import (
    Block,
    Br,
    Cbr,
    Instr,
    KernelIR,
    Op,
    Param,
    Ret,
    SharedDecl,
    VReg,
)
from ptxlab.validate import validate


def lower(kernel: Kernel) -> KernelIR:
    lowerer = _Lowerer(kernel)
    lowerer.run()
    ir = KernelIR(
        name=kernel.name,
        params=[Param(param.name, param.dtype, param.shape) for param in kernel.params],
        shared=[SharedDecl(item.name, item.dtype, item.shape) for item in kernel.shared],
        blocks=lowerer.blocks,
        version=kernel.version,
        target=kernel.target,
        address_size=kernel.address_size,
        expect_block=kernel.expect_block,
    )
    validate(ir)
    return ir


class _Lowerer:
    def __init__(self, kernel: Kernel) -> None:
        self.kernel = kernel
        self.blocks: list[Block] = []
        self.cur: Block | None = None
        self._tmp_id = 0
        self._label = 0
        self._names: set[str] = set()
        self._locals: dict[str, DType] = {}
        self.arrays = {
            param.name: ArrayRef(param.name, param.dtype, param.shape)
            for param in kernel.params
            if param.shape
        }
        self.shared = {item.name: SharedRef(item.name, item.dtype, item.shape) for item in kernel.shared}
        self.element_size = {
            **{name: ref.dtype.size_bytes for name, ref in self.arrays.items()},
            **{name: ref.dtype.size_bytes for name, ref in self.shared.items()},
        }

    def run(self) -> None:
        self._start("entry")
        for stmt in self.kernel.body:
            self._stmt(stmt)
        if self.cur is None or self.cur.term is not None:
            raise LoweringError("kernel body did not leave an open block")
        self.cur.term = Ret()

    def _stmt(self, stmt: object) -> None:
        if isinstance(stmt, AssignStmt):
            self._expr(stmt.expr, self._local(stmt.name, stmt.dtype))
            return
        if isinstance(stmt, StoreStmt):
            array = self.arrays[stmt.array]
            linear = self._linear(array.shape, stmt.indices)
            addr = self._global_addr(array.name, linear, array.dtype)
            value = self._expr(stmt.value)
            self._emit(Instr(Op.ST_GLOBAL, None, (addr, value)))
            return
        if isinstance(stmt, StoreSharedStmt):
            buf = self.shared[stmt.buffer]
            addr = self._shared_addr(buf.name, self._linear(buf.shape, stmt.indices), buf.dtype)
            value = self._expr(stmt.value)
            self._emit(Instr(Op.ST_SHARED, None, (addr, value)))
            return
        if isinstance(stmt, IfStmt):
            pred = self._expr(stmt.cond)
            if pred.dtype is not DType.PRED:
                raise LoweringError("if condition must be a predicate")
            then_l = self._fresh_label("then")
            else_l = self._fresh_label("else")
            join_l = self._fresh_label("join")
            self._seal(Cbr(pred, then_l, else_l))
            self._start(then_l)
            for item in stmt.then_body:
                self._stmt(item)
            self._seal(Br(join_l))
            self._start(else_l)
            for item in stmt.else_body:
                self._stmt(item)
            self._seal(Br(join_l))
            self._start(join_l)
            return
        if isinstance(stmt, ForStmt):
            var = self._local(stmt.var, DType.U32)
            self._expr(stmt.start, var)
            header = self._fresh_label("for")
            body = self._fresh_label("for_body")
            exit_l = self._fresh_label("for_exit")
            self._seal(Br(header))
            self._start(header)
            end = self._expr(stmt.end)
            pred = self._tmp(DType.PRED)
            self._emit(Instr(Op.CMP, pred, (var, end), dtype=DType.U32, cmp="lt"))
            self._seal(Cbr(pred, body, exit_l))
            self._start(body)
            for item in stmt.body:
                self._stmt(item)
            step = self._expr(stmt.step)
            self._emit(Instr(Op.ADD, var, (var, step), dtype=DType.U32))
            self._seal(Br(header))
            self._start(exit_l)
            return
        if isinstance(stmt, BarrierStmt):
            self._emit(Instr(Op.BAR, None, (stmt.bar_id,)))
            return
        raise LoweringError(f"unknown statement {type(stmt).__name__}")

    def _expr(self, expr: Expr, dest: VReg | None = None) -> VReg:
        produced = self._produce(expr)
        if dest is None or dest.name == produced.name:
            return produced
        if dest.dtype is not produced.dtype:
            raise LoweringError(f"cannot move {produced.dtype.value} into {dest.dtype.value}")
        self._emit(Instr(Op.MOV, dest, (produced,)))
        return dest

    def _produce(self, expr: Expr) -> VReg:
        if isinstance(expr, ConstExpr):
            dest = self._tmp(expr.dtype)
            self._emit(Instr(Op.CONST, dest, (expr.value,), dtype=expr.dtype))
            return dest
        if isinstance(expr, VarExpr):
            return self._local(expr.name, expr.dtype)
        if isinstance(expr, ScalarExpr):
            dest = self._tmp(expr.dtype)
            self._emit(Instr(Op.LD_PARAM, dest, symbol=expr.name))
            return dest
        if isinstance(expr, BinExpr):
            lhs = self._expr(expr.left)
            rhs = self._expr(expr.right)
            dest = self._tmp(expr.dtype)
            op = {"add": Op.ADD, "sub": Op.SUB, "mul": Op.MUL}[expr.op]
            self._emit(Instr(op, dest, (lhs, rhs), dtype=expr.dtype))
            return dest
        if isinstance(expr, CmpExpr):
            lhs = self._expr(expr.left)
            rhs = self._expr(expr.right)
            dest = self._tmp(DType.PRED)
            self._emit(Instr(Op.CMP, dest, (lhs, rhs), dtype=expr.operand_dtype, cmp=expr.op))
            return dest
        if isinstance(expr, PredExpr):
            dest = self._tmp(DType.PRED)
            if expr.op == "not":
                src = self._expr(expr.args[0])
                self._emit(Instr(Op.NOT_PRED, dest, (src,)))
                return dest
            lhs = self._expr(expr.args[0])
            rhs = self._expr(expr.args[1])
            op = Op.AND_PRED if expr.op == "and" else Op.OR_PRED
            self._emit(Instr(op, dest, (lhs, rhs)))
            return dest
        if isinstance(expr, IndexExpr):
            dest = self._tmp(DType.U32)
            op = {"tid": Op.TID, "ctaid": Op.CTAID, "ntid": Op.NTID}[expr.kind]
            self._emit(Instr(op, dest, dim=expr.dim))
            return dest
        if isinstance(expr, LoadExpr):
            array = self.arrays[expr.array]
            linear = self._linear(array.shape, expr.indices)
            addr = self._global_addr(array.name, linear, array.dtype)
            dest = self._tmp(array.dtype)
            self._emit(Instr(Op.LD_GLOBAL, dest, (addr,)))
            return dest
        if isinstance(expr, LoadSharedExpr):
            buf = self.shared[expr.buffer]
            addr = self._shared_addr(buf.name, self._linear(buf.shape, expr.indices), buf.dtype)
            dest = self._tmp(buf.dtype)
            self._emit(Instr(Op.LD_SHARED, dest, (addr,)))
            return dest
        if isinstance(expr, MadExpr):
            a = self._expr(expr.a)
            b = self._expr(expr.b)
            c = self._expr(expr.c)
            dest = self._tmp(expr.dtype)
            self._emit(Instr(Op.MAD_LO, dest, (a, b, c), dtype=expr.dtype))
            return dest
        if isinstance(expr, ShrExpr):
            value = self._expr(expr.value)
            amount = self._expr(expr.amount)
            dest = self._tmp(DType.U32)
            self._emit(Instr(Op.SHR, dest, (value, amount), dtype=DType.U32))
            return dest
        raise LoweringError(f"unknown expression {type(expr).__name__}")

    def _linear(self, shape: tuple, indices: tuple[Expr, ...]) -> VReg:
        acc: VReg | None = None
        for position, index in enumerate(indices):
            idx = self._expr(index)
            trailing = shape[position + 1 :]
            term = idx if not trailing else self._mul(idx, self._product(trailing))
            acc = term if acc is None else self._add_u32(acc, term)
        if acc is None:
            raise LoweringError("missing index")
        return acc

    def _product(self, dims: tuple) -> VReg:
        acc = self._const_u32(1)
        for dim in dims:
            if getattr(dim, "value", None) is not None or isinstance(dim, int):
                value = dim if isinstance(dim, int) else dim.value
                acc = self._mul(acc, self._const_u32(value))
            else:
                reg = self._tmp(DType.U32)
                self._emit(Instr(Op.LD_PARAM, reg, symbol=dim.name))
                acc = self._mul(acc, reg)
        return acc

    def _global_addr(self, name: str, linear: VReg, dtype: DType) -> VReg:
        base = self._tmp(DType.U64)
        self._emit(Instr(Op.LD_PARAM, base, symbol=name))
        return self._offset_addr(base, linear, dtype.size_bytes, wide=True)

    def _shared_addr(self, name: str, linear: VReg, dtype: DType) -> VReg:
        base = self._tmp(DType.U32)
        self._emit(Instr(Op.SHARED_BASE, base, symbol=name))
        if dtype.size_bytes == 4:
            byte = self._mul(linear, self._const_u32(4))
        else:
            raise LoweringError("shared elements must be 4 bytes in this subset")
        return self._add_u32(base, byte)

    def _offset_addr(self, base: VReg, linear: VReg, nbytes: int, *, wide: bool) -> VReg:
        if nbytes <= 0 or nbytes > 255:
            raise LoweringError("unsupported element size")
        size = self._const_u32(nbytes)
        wide_reg = self._tmp(DType.U64)
        self._emit(Instr(Op.MUL_WIDE, wide_reg, (linear, size)))
        dest = self._tmp(DType.U64)
        self._emit(Instr(Op.ADD, dest, (base, wide_reg), dtype=DType.U64))
        return dest

    def _mul(self, lhs: VReg, rhs: VReg) -> VReg:
        dest = self._tmp(DType.U32)
        self._emit(Instr(Op.MUL, dest, (lhs, rhs), dtype=DType.U32))
        return dest

    def _add_u32(self, lhs: VReg, rhs: VReg) -> VReg:
        dest = self._tmp(DType.U32)
        self._emit(Instr(Op.ADD, dest, (lhs, rhs), dtype=DType.U32))
        return dest

    def _const_u32(self, value: int) -> VReg:
        dest = self._tmp(DType.U32)
        self._emit(Instr(Op.CONST, dest, (value,), dtype=DType.U32))
        return dest

    def _local(self, name: str, dtype: DType) -> VReg:
        previous = self._locals.get(name)
        if previous is not None and previous is not dtype:
            raise LoweringError(f"{name} is both {previous.value} and {dtype.value}")
        reg = VReg(f"%{name}", dtype)
        if previous is None and reg.name in self._names:
            raise LoweringError(f"name collision on {name}")
        self._locals[name] = dtype
        self._names.add(reg.name)
        return reg

    def _tmp(self, dtype: DType) -> VReg:
        reg = VReg(f"%t{self._tmp_id}", dtype)
        self._tmp_id += 1
        if reg.name in self._names:
            raise LoweringError(f"temporary {reg.name} collided with a local")
        self._names.add(reg.name)
        return reg

    def _fresh_label(self, stem: str) -> str:
        label = f"L_{stem}_{self._label}"
        self._label += 1
        return label

    def _start(self, label: str) -> None:
        if self.cur is not None and self.cur.term is None:
            raise LoweringError(f"block {self.cur.label} is missing a terminator")
        block = Block(label)
        self.blocks.append(block)
        self.cur = block

    def _seal(self, term: Br | Cbr | Ret) -> None:
        if self.cur is None or self.cur.term is not None:
            raise LoweringError("no open block to terminate")
        self.cur.term = term

    def _emit(self, instr: Instr) -> None:
        if self.cur is None or self.cur.term is not None:
            raise LoweringError("instruction outside an open block")
        self.cur.instrs.append(instr)
