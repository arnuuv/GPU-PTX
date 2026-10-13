"""Kernel builders for the supported examples.

Reduction and tiled matrix multiplication use shared memory and ``bar.sync``.
The reduction block size is a compile-time power of two because the shared
array and the tree offsets are static. Tiled GEMM uses a compile-time square tile.
"""

from __future__ import annotations

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
