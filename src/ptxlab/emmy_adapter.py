"""Isolated proposal for an Emmy-facing PTX adapter.

This module does not import Emmy and does not call Emmy's compiler. ``EmmyKernelDescriptor``
is a PTXLab type. It is not ``CudaOp``, ``KernelOp``, or any other Emmy class.

Emmy's CUDA lowering renders a ``__global__`` CUDA C string and ``nvcc`` turns that
into a cubin. Treating that string as PTX would be wrong. A real adapter would
have to recognise a narrow kernel subset and lower it through PTXLab, then sit
beside ``backend/cuda`` rather than replacing ``kernel_source``. That recogniser
is not implemented. See ``docs/emmy-integration.md``.
"""

from __future__ import annotations

from dataclasses import dataclass

from ptxlab.driver import compile_kernel, execute_kernel
from ptxlab.dsl import Kernel
from ptxlab.errors import LaunchError
from ptxlab.sim import SimResult

ASSUMPTIONS: tuple[str, ...] = (
    "Emmy was inspected as source. This adapter was not run against an installed Emmy package.",
    "Emmy CudaOp.kernel_source is CUDA C, compiled by nvcc to a cubin. It is not PTX text.",
    "Inline asm in Emmy's CUDA renderer (mma, ldmatrix, cp.async, mbarrier) is not a PTX module.",
    "PTXLab only lowers its own AST. It does not accept an Emmy Graph or KernelOp.",
    "Symbolic grid extents, TMA descriptors, and the cubin cache are outside this interface.",
    "simulate() runs the CPU interpreter. It does not launch a GPU and does not report a speedup.",
)


@dataclass(frozen=True)
class EmmyKernelDescriptor:
    """Launch geometry plus a PTXLab kernel.

    ``name`` must match the kernel. ``block`` must match ``kernel.expect_block``
    when the kernel fixes the block size, which reduction and tiled GEMM do.
    """

    name: str
    kernel: Kernel
    grid: tuple[int, int, int]
    block: tuple[int, int, int]


class EmmyPtxAdapter:
    """Compile and simulate a descriptor. The PTX is produced by PTXLab's emitter."""

    assumptions = ASSUMPTIONS

    def emit_ptx(self, descriptor: EmmyKernelDescriptor) -> str:
        self._check(descriptor)
        return compile_kernel(descriptor.kernel)

    def simulate(self, descriptor: EmmyKernelDescriptor, args: dict[str, object]) -> SimResult:
        self._check(descriptor)
        return execute_kernel(
            descriptor.kernel,
            grid=descriptor.grid,
            block=descriptor.block,
            args=args,
        )

    def _check(self, descriptor: EmmyKernelDescriptor) -> None:
        if descriptor.kernel.name != descriptor.name:
            raise LaunchError(
                f"descriptor name {descriptor.name} does not match kernel {descriptor.kernel.name}"
            )
        if descriptor.kernel.expect_block is not None and descriptor.block != descriptor.kernel.expect_block:
            raise LaunchError(
                f"kernel expects block {descriptor.kernel.expect_block}, descriptor has {descriptor.block}"
            )
