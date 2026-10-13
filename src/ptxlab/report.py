"""Assemble a view of one kernel for the dashboard and for tests."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

from ptxlab.driver import compile_kernel, execute_kernel
from ptxlab.dump import dump_ir
from ptxlab.lower import lower
from ptxlab.parse import parse
from ptxlab.perf import estimate


def report_from_path(path: Path) -> dict[str, object]:
    module = _load(path)
    kernel = module.build_kernel()
    grid, block, args, compare = module.sample()
    ir = lower(kernel)
    ptx = compile_kernel(kernel)
    program = parse(ptx)
    result = execute_kernel(kernel, grid=grid, block=block, args=args, trace=True)
    perf = estimate(program, result, grid=_pad(grid), block=_pad(block))
    errors: dict[str, float] = {}
    for name, expected in compare.items():
        actual = args[name]
        if isinstance(actual, np.ndarray) and actual.dtype == np.float32:
            errors[name] = float(np.max(np.abs(actual.astype(np.float64) - expected.astype(np.float64))))
    return {
        "name": program.name,
        "ir": dump_ir(ir),
        "ptx": ptx,
        "inst_counts": result.inst_counts,
        "divergent_branches": result.divergent_branches,
        "instructions": result.instructions,
        "memory_transactions": perf.memory_transactions,
        "estimated_cycles": perf.estimated_cycles,
        "occupancy": perf.occupancy,
        "utilization": perf.utilization,
        "assumptions": list(perf.assumptions),
        "errors": errors,
        "timeline": [
            {
                "block": list(access.block),
                "warp": access.warp,
                "pc": access.pc,
                "op": access.op,
                "lanes": len(access.addrs),
                "first_addr": access.addrs[0] if access.addrs else None,
            }
            for access in result.mem_accesses[:24]
        ],
        "trace": [
            {
                "block": list(event.block),
                "warp": event.warp,
                "pc": event.pc,
                "op": event.op,
                "active_mask": event.active_mask,
            }
            for event in result.trace[:32]
        ],
    }


def _load(path: Path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    if spec is None or spec.loader is None:
        raise FileNotFoundError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _pad(values: tuple[int, ...]) -> tuple[int, int, int]:
    dims = [1, 1, 1]
    for index, value in enumerate(values[:3]):
        dims[index] = int(value)
    return dims[0], dims[1], dims[2]
