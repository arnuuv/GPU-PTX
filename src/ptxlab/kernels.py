"""Kernel builders for the supported examples.

Reduction and tiled matrix multiplication use shared memory and ``bar.sync``.
The reduction block size is a compile-time power of two because the shared
array and the tree offsets are static. Tiled GEMM uses a compile-time square tile.
"""

from __future__ import annotations

import numpy as np

from ptxlab.dsl import Kernel, KernelBuilder
from ptxlab.dtype import DType


def vector_add() -> Kernel:
    k = KernelBuilder("vector_add")
    a = k.array("a", DType.F32, ("n",))
    b = k.array("b", DType.F32, ("n",))
    c = k.array("c", DType.F32, ("n",))
    n = k.scalar("n", DType.U32)
    i = k.let("i", k.gid("x"))
    with k.when(i < n):
        k.store(c, (i,), k.load(a, (i,)) + k.load(b, (i,)))
    return k.build()


def vector_mul() -> Kernel:
    k = KernelBuilder("vector_mul")
    a = k.array("a", DType.F32, ("n",))
    b = k.array("b", DType.F32, ("n",))
    c = k.array("c", DType.F32, ("n",))
    n = k.scalar("n", DType.U32)
    i = k.let("i", k.gid("x"))
    with k.when(i < n):
        k.store(c, (i,), k.load(a, (i,)) * k.load(b, (i,)))
    return k.build()


def vector_add_f16() -> Kernel:
    k = KernelBuilder("vector_add_f16")
    a = k.array("a", DType.F16, ("n",))
    b = k.array("b", DType.F16, ("n",))
    c = k.array("c", DType.F16, ("n",))
    n = k.scalar("n", DType.U32)
    i = k.let("i", k.gid("x"))
    with k.when(i < n):
        total = k.to_f32(k.load(a, (i,))) + k.to_f32(k.load(b, (i,)))
        k.store(c, (i,), k.to_f16(total))
    return k.build()


def vector_add_v4() -> Kernel:
    """One thread owns four consecutive f32 values. ``nvec`` is ``n / 4``."""

    k = KernelBuilder("vector_add_v4")
    a = k.array("a", DType.F32, ("n",))
    b = k.array("b", DType.F32, ("n",))
    c = k.array("c", DType.F32, ("n",))
    k.scalar("n", DType.U32)
    nvec = k.scalar("nvec", DType.U32)
    i = k.let("i", k.gid("x"))
    with k.when(i < nvec):
        a0, a1, a2, a3 = k.load_v4(a, i)
        b0, b1, b2, b3 = k.load_v4(b, i)
        k.store_v4(c, i, (a0 + b0, a1 + b1, a2 + b2, a3 + b3))
    return k.build()


def fma_saxpy() -> Kernel:
    k = KernelBuilder("fma_saxpy")
    x = k.array("x", DType.F32, ("n",))
    y = k.array("y", DType.F32, ("n",))
    z = k.array("z", DType.F32, ("n",))
    alpha = k.scalar("alpha", DType.F32)
    n = k.scalar("n", DType.U32)
    i = k.let("i", k.gid("x"))
    with k.when(i < n):
        k.store(z, (i,), k.fma(alpha, k.load(x, (i,)), k.load(y, (i,))))
    return k.build()


def atomic_count() -> Kernel:
    k = KernelBuilder("atomic_count")
    counter = k.array("counter", DType.U32, (1,))
    n = k.scalar("n", DType.U32)
    i = k.let("i", k.gid("x"))
    with k.when(i < n):
        k.let("old", k.atomic_add(counter, (0,), k.u32(1)))
    return k.build()


def saxpy() -> Kernel:
    k = KernelBuilder("saxpy")
    x = k.array("x", DType.F32, ("n",))
    y = k.array("y", DType.F32, ("n",))
    z = k.array("z", DType.F32, ("n",))
    alpha = k.scalar("alpha", DType.F32)
    n = k.scalar("n", DType.U32)
    i = k.let("i", k.gid("x"))
    with k.when(i < n):
        k.store(z, (i,), alpha * k.load(x, (i,)) + k.load(y, (i,)))
    return k.build()


def reduction(block_size: int) -> Kernel:
    if block_size < 1 or block_size > 1024 or block_size & (block_size - 1):
        raise ValueError("reduction block size must be a power of two in 1..1024")
    k = KernelBuilder("reduce", expect_block=(block_size, 1, 1))
    src = k.array("src", DType.F32, ("n",))
    dst = k.array("dst", DType.F32, ("blocks",))
    n = k.scalar("n", DType.U32)
    blocks = k.scalar("blocks", DType.U32)
    smem = k.shared("sdata", DType.F32, (block_size,))
    tx = k.let("tx", k.tid("x"))
    gid = k.let("gid", k.ctaid("x") * k.u32(block_size) + tx)
    with k.when(gid < n) as inbound:
        k.store_shared(smem, (tx,), k.load(src, (gid,)))
    with inbound.otherwise():
        k.store_shared(smem, (tx,), k.f32(0))
    k.barrier()
    offset = block_size // 2
    while offset:
        with k.when(tx < k.u32(offset)):
            total = k.load_shared(smem, (tx,)) + k.load_shared(smem, (tx + k.u32(offset),))
            k.store_shared(smem, (tx,), total)
        k.barrier()
        offset //= 2
    with k.when(k.land(tx == k.u32(0), k.ctaid("x") < blocks)):
        k.store(dst, (k.ctaid("x"),), k.load_shared(smem, (0,)))
    return k.build()


def tiled_matmul(tile: int = 16) -> Kernel:
    if tile < 1 or tile > 32:
        raise ValueError("tile must be in 1..32 so the block stays within 1024 threads")
    k = KernelBuilder("matmul", expect_block=(tile, tile, 1))
    left = k.array("a", DType.F32, ("m", "k"))
    right = k.array("b", DType.F32, ("k", "n"))
    out = k.array("c", DType.F32, ("m", "n"))
    rows = k.scalar("m", DType.U32)
    cols = k.scalar("n", DType.U32)
    depth = k.scalar("k", DType.U32)
    tile_a = k.shared("asmem", DType.F32, (tile, tile))
    tile_b = k.shared("bsmem", DType.F32, (tile, tile))
    tx = k.let("tx", k.tid("x"))
    ty = k.let("ty", k.tid("y"))
    row = k.let("row", k.ctaid("y") * k.u32(tile) + ty)
    col = k.let("col", k.ctaid("x") * k.u32(tile) + tx)
    acc = k.let("acc", k.f32(0))
    with k.for_range("kt", 0, depth, tile) as kt:
        a_col = k.let("acol", kt + tx)
        b_row = k.let("brow", kt + ty)
        with k.when(k.land(row < rows, a_col < depth)) as a_in:
            k.store_shared(tile_a, (ty, tx), k.load(left, (row, a_col)))
        with a_in.otherwise():
            k.store_shared(tile_a, (ty, tx), k.f32(0))
        with k.when(k.land(b_row < depth, col < cols)) as b_in:
            k.store_shared(tile_b, (ty, tx), k.load(right, (b_row, col)))
        with b_in.otherwise():
            k.store_shared(tile_b, (ty, tx), k.f32(0))
        k.barrier()
        with k.for_range("inner", 0, tile) as inner:
            product = k.load_shared(tile_a, (ty, inner)) * k.load_shared(tile_b, (inner, tx))
            k.assign(acc, acc + product)
        k.barrier()
    with k.when(k.land(row < rows, col < cols)):
        k.store(out, (row, col), acc)
    return k.build()


def _require_block(block_size: int) -> None:
    if block_size < 1 or block_size > 1024 or block_size & (block_size - 1):
        raise ValueError("block size must be a power of two in 1..1024")


def _tree(k: KernelBuilder, smem, tx, block_size: int, combine) -> None:
    offset = block_size // 2
    while offset:
        with k.when(tx < k.u32(offset)):
            left = k.load_shared(smem, (tx,))
            right = k.load_shared(smem, (tx + k.u32(offset),))
            k.store_shared(smem, (tx,), combine(left, right))
        k.barrier()
        offset //= 2


def layer_norm(block_size: int) -> Kernel:
    """One block normalises one row of ``block_size`` f32 values."""

    _require_block(block_size)
    k = KernelBuilder("layer_norm", expect_block=(block_size, 1, 1))
    x = k.array("x", DType.F32, ("rows", block_size))
    gamma = k.array("gamma", DType.F32, (block_size,))
    beta = k.array("beta", DType.F32, (block_size,))
    y = k.array("y", DType.F32, ("rows", block_size))
    rows = k.scalar("rows", DType.U32)
    eps = k.scalar("eps", DType.F32)
    smem = k.shared("sm", DType.F32, (block_size,))
    tx = k.let("tx", k.tid("x"))
    row = k.let("row", k.ctaid("x"))
    xv = k.let("xv", k.f32(0))
    with k.when(row < rows):
        k.assign(xv, k.load(x, (row, tx)))
    k.store_shared(smem, (tx,), xv)
    k.barrier()
    _tree(k, smem, tx, block_size, lambda left, right: left + right)
    mean = k.let("mean", k.div(k.load_shared(smem, (0,)), k.f32(float(block_size))))
    centered = k.let("centered", xv - mean)
    k.store_shared(smem, (tx,), centered * centered)
    k.barrier()
    _tree(k, smem, tx, block_size, lambda left, right: left + right)
    inv = k.let("inv", k.div(k.f32(1), k.sqrt(k.div(k.load_shared(smem, (0,)), k.f32(float(block_size))) + eps)))
    with k.when(row < rows):
        scale = centered * inv * k.load(gamma, (tx,))
        k.store(y, (row, tx), scale + k.load(beta, (tx,)))
    return k.build()


def rms_norm(block_size: int) -> Kernel:
    _require_block(block_size)
    k = KernelBuilder("rms_norm", expect_block=(block_size, 1, 1))
    x = k.array("x", DType.F32, ("rows", block_size))
    weight = k.array("weight", DType.F32, (block_size,))
    y = k.array("y", DType.F32, ("rows", block_size))
    rows = k.scalar("rows", DType.U32)
    eps = k.scalar("eps", DType.F32)
    smem = k.shared("sm", DType.F32, (block_size,))
    tx = k.let("tx", k.tid("x"))
    row = k.let("row", k.ctaid("x"))
    xv = k.let("xv", k.f32(0))
    with k.when(row < rows):
        k.assign(xv, k.load(x, (row, tx)))
    k.store_shared(smem, (tx,), xv * xv)
    k.barrier()
    _tree(k, smem, tx, block_size, lambda left, right: left + right)
    inv = k.let("inv", k.div(k.f32(1), k.sqrt(k.div(k.load_shared(smem, (0,)), k.f32(float(block_size))) + eps)))
    with k.when(row < rows):
        k.store(y, (row, tx), xv * inv * k.load(weight, (tx,)))
    return k.build()


def softmax(block_size: int) -> Kernel:
    """Row softmax. ``ex2.approx.ftz.f32`` is evaluated as flushed ``exp2`` in the interpreter."""

    _require_block(block_size)
    log2e = float(np.float32(np.log2(np.e)))
    k = KernelBuilder("softmax", expect_block=(block_size, 1, 1))
    x = k.array("x", DType.F32, ("rows", block_size))
    y = k.array("y", DType.F32, ("rows", block_size))
    rows = k.scalar("rows", DType.U32)
    smem = k.shared("sm", DType.F32, (block_size,))
    tx = k.let("tx", k.tid("x"))
    row = k.let("row", k.ctaid("x"))
    xv = k.let("xv", k.f32(0))
    with k.when(row < rows):
        k.assign(xv, k.load(x, (row, tx)))
    k.store_shared(smem, (tx,), xv)
    k.barrier()
    _tree(k, smem, tx, block_size, lambda left, right: k.maximum(left, right))
    peak = k.let("peak", k.load_shared(smem, (0,)))
    expo = k.let("expo", k.ex2((xv - peak) * k.f32(log2e)))
    k.store_shared(smem, (tx,), expo)
    k.barrier()
    _tree(k, smem, tx, block_size, lambda left, right: left + right)
    with k.when(row < rows):
        k.store(y, (row, tx), k.div(expo, k.load_shared(smem, (0,))))
    return k.build()
