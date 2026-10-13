"""Compile a kernel AST and run it on the functional simulator."""

from __future__ import annotations

import numpy as np

from ptxlab.dsl import Kernel
from ptxlab.dtype import DType
from ptxlab.emit import emit
from ptxlab.errors import LaunchError
from ptxlab.lower import lower
from ptxlab.opt import OptimizeConfig, optimize
from ptxlab.parse import parse
from ptxlab.sim import SimResult, simulate


def compile_kernel(
    kernel: Kernel,
    *,
    config: OptimizeConfig | None = None,
    allocation: str = "greedy",
) -> str:
    ir, _notes = optimize(lower(kernel), config if config is not None else OptimizeConfig())
    return emit(ir, allocation=allocation)


def execute_kernel(
    kernel: Kernel,
    *,
    grid: tuple[int, ...] | list[int],
    block: tuple[int, ...] | list[int],
    args: dict[str, object],
    trace: bool = False,
) -> SimResult:
    check_shapes(kernel, args)
    program = parse(compile_kernel(kernel))
    return simulate(program, grid=grid, block=block, args=args, trace=trace)


def check_shapes(kernel: Kernel, args: dict[str, object]) -> None:
    """Require each pointer allocation to match the kernel's declared shape."""

    for param in kernel.params:
        if not param.shape:
            continue
        value = args.get(param.name)
        if not isinstance(value, np.ndarray):
            raise LaunchError(f"{param.name} expects an array")
        extent = 1
        for dim in param.shape:
            if dim.value is not None:
                extent *= dim.value
            else:
                assert dim.name is not None
                if dim.name not in args:
                    raise LaunchError(f"shape of {param.name} depends on missing scalar {dim.name}")
                scalar = args[dim.name]
                if isinstance(scalar, bool) or not isinstance(scalar, (int, np.integer)):
                    raise LaunchError(f"extent {dim.name} must be an integer")
                if int(scalar) < 0:
                    raise LaunchError(f"extent {dim.name} is negative")
                extent *= int(scalar)
        if value.size != extent:
            raise LaunchError(f"{param.name} has {value.size} elements, shape expects {extent}")
        if param.dtype is DType.F32 and value.dtype != np.float32:
            raise LaunchError(f"{param.name} expects float32")
        if param.dtype is DType.S32 and value.dtype != np.int32:
            raise LaunchError(f"{param.name} expects int32")
        if param.dtype is DType.U32 and value.dtype != np.uint32:
            raise LaunchError(f"{param.name} expects uint32")
