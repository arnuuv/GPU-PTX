"""Analytical performance model."""

import numpy as np

from ptxlab.driver import compile_kernel, execute_kernel
from ptxlab.dtype import DType
from ptxlab.dsl import KernelBuilder
from ptxlab.kernels import vector_add
from ptxlab.parse import parse
from ptxlab.perf import ArchConfig, estimate, occupancy


def test_occupancy_limits() -> None:
    blocks, occ = occupancy(threads_per_block=256, regs_per_thread=32, shared_bytes=1024)
    assert blocks == 8
    assert occ == 1.0
    blocks, occ = occupancy(threads_per_block=256, regs_per_thread=255, shared_bytes=0)
    assert blocks == 1
    assert occ == 256 / 2048


def test_coalesced_vector_add_uses_fewer_sectors_than_a_stride() -> None:
    n = 32
    a = np.arange(n, dtype=np.float32)
    b = np.arange(n, dtype=np.float32)
    c = np.zeros(n, np.float32)
    program = parse(compile_kernel(vector_add()))
    result = execute_kernel(
        vector_add(),
        grid=(1, 1, 1),
        block=(32, 1, 1),
        args={"a": a, "b": b, "c": c, "n": n},
    )
    coalesced = estimate(program, result, grid=(1, 1, 1), block=(32, 1, 1))

    stride = KernelBuilder("stride")
    src = stride.array("src", DType.F32, ("n",))
    dst = stride.array("dst", DType.F32, ("n",))
    length = stride.scalar("n", DType.U32)
    i = stride.let("i", stride.gid("x"))
    with stride.when(i < length):
        index = i * stride.u32(32)
        stride.store(dst, (index,), stride.load(src, (index,)))
    wide = 32 * 32
    src_a = np.arange(wide, dtype=np.float32)
    dst_a = np.zeros(wide, np.float32)
    kernel = stride.build()
    scattered_program = parse(compile_kernel(kernel))
    scattered = execute_kernel(
        kernel,
        grid=(1, 1, 1),
        block=(32, 1, 1),
        args={"src": src_a, "dst": dst_a, "n": wide},
    )
    scattered_est = estimate(scattered_program, scattered, grid=(1, 1, 1), block=(32, 1, 1))
    assert coalesced.memory_transactions < scattered_est.memory_transactions
    assert coalesced.estimated_cycles is not None
    assert scattered_est.estimated_cycles is not None
    assert scattered_est.estimated_cycles > coalesced.estimated_cycles
    assert "not a measured GPU time" in coalesced.assumptions[0]
    assert coalesced.kind == "analytical estimate"


def test_custom_sm_count_changes_waves() -> None:
    n = 32
    a = np.ones(n, np.float32)
    b = np.ones(n, np.float32)
    c = np.zeros(n, np.float32)
    kernel = vector_add()
    program = parse(compile_kernel(kernel))
    result = execute_kernel(kernel, grid=(4, 1, 1), block=(32, 1, 1), args={"a": a, "b": b, "c": c, "n": n})
    one = estimate(program, result, grid=(4, 1, 1), block=(32, 1, 1), arch=ArchConfig(num_sms=1, max_blocks_per_sm=1))
    many = estimate(program, result, grid=(4, 1, 1), block=(32, 1, 1), arch=ArchConfig(num_sms=4, max_blocks_per_sm=1))
    assert one.waves == 4
    assert many.waves == 1
    assert one.estimated_cycles is not None and many.estimated_cycles is not None
    assert one.estimated_cycles > many.estimated_cycles
