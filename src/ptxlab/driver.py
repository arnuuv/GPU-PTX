"""Compile a kernel AST and run it on the functional simulator."""

from __future__ import annotations

from ptxlab.dsl import Kernel
from ptxlab.emit import emit
from ptxlab.lower import lower
from ptxlab.parse import parse
from ptxlab.sim import SimResult, simulate


def compile_kernel(kernel: Kernel) -> str:
    return emit(lower(kernel))


def execute_kernel(
    kernel: Kernel,
    *,
    grid: tuple[int, ...] | list[int],
    block: tuple[int, ...] | list[int],
    args: dict[str, object],
    trace: bool = False,
) -> SimResult:
    program = parse(compile_kernel(kernel))
    return simulate(program, grid=grid, block=block, args=args, trace=trace)
