"""Command line: compile, simulate, test, bench, dashboard."""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np

from ptxlab.driver import compile_kernel, execute_kernel
from ptxlab.launchio import dump_launch, load_launch
from ptxlab.parse import parse
from ptxlab.perf import estimate
from ptxlab.sim import simulate


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="ptxlab", description="Compile and simulate a small NVIDIA PTX subset.")
    sub = parser.add_subparsers(dest="command", required=True)

    compile_p = sub.add_parser("compile", help="compile a Python kernel module to PTX")
    compile_p.add_argument("source", type=Path)
    compile_p.add_argument("-o", "--output", type=Path)

    simulate_p = sub.add_parser("simulate", help="run an emitted PTX file on the CPU simulator")
    simulate_p.add_argument("ptx", type=Path)
    simulate_p.add_argument("--launch", type=Path, help="launch JSON; defaults to the .launch.json beside the PTX")

    sub.add_parser("test", help="run the pytest suite")
    sub.add_parser("bench", help="run the analytical benchmark script")
    dash = sub.add_parser("dashboard", help="open the local Streamlit viewer")
    dash.add_argument("--port", type=int, default=8501)

    args = parser.parse_args(argv)
    if args.command == "compile":
        return _compile(args.source, args.output)
    if args.command == "simulate":
        return _simulate(args.ptx, args.launch)
    if args.command == "test":
        return _test()
    if args.command == "bench":
        return _bench()
    if args.command == "dashboard":
        return _dashboard(args.port)
    parser.error(f"unknown command {args.command}")
    return 2


def _compile(source: Path, output: Path | None) -> int:
    module = _load_module(source)
    kernel = module.build_kernel()
    text = compile_kernel(kernel)
    destination = output or source.with_suffix(".ptx")
    destination.write_text(text, encoding="utf-8")
    if hasattr(module, "sample"):
        grid, block, args, compare = module.sample()
        dump_launch(destination.with_suffix(".launch.json"), grid=grid, block=block, args=args, compare=compare)
    print(f"wrote {destination}")
    return 0


def _simulate(ptx_path: Path, launch_path: Path | None) -> int:
    program = parse(ptx_path.read_text(encoding="utf-8"))
    launch = launch_path or ptx_path.with_suffix(".launch.json")
    if not launch.exists():
        print(f"no launch file at {launch}", file=sys.stderr)
        return 1
    grid, block, args, compare = load_launch(launch)
    result = simulate(program, grid=grid, block=block, args=args, trace=False)
    perf = estimate(program, result, grid=_pad(grid), block=_pad(block))
    print(f"kernel {program.name}")
    print(f"instructions {result.instructions}  divergent_branches {result.divergent_branches}")
    print(f"global_transactions {perf.memory_transactions}  occupancy {perf.occupancy:.3f}")
    if perf.estimated_cycles is None:
        print("estimated_cycles unavailable (kernel does not fit the model)")
    else:
        print(f"estimated_cycles {perf.estimated_cycles}  (analytical estimate, not a GPU measurement)")
    worst = 0.0
    for name, expected in compare.items():
        actual = args[name]
        if not isinstance(actual, np.ndarray):
            continue
        if actual.dtype == np.float32:
            error = float(np.max(np.abs(actual.astype(np.float64) - expected.astype(np.float64))))
            worst = max(worst, error)
            print(f"{name} max_abs_error {error:.3e}")
            if error != 0.0:
                return 1
        else:
            mismatches = int(np.count_nonzero(actual != expected))
            print(f"{name} mismatches {mismatches}")
            if mismatches:
                return 1
    if compare and worst == 0.0:
        print("reference match: exact on floating outputs")
    return 0


def _test() -> int:
    try:
        import pytest
    except ImportError:
        print("pytest is not installed; install ptxlab with the dev extra", file=sys.stderr)
        return 1
    return int(pytest.main(["-q", "tests"]))


def _bench() -> int:
    root = Path(__file__).resolve().parents[2]
    script = root / "benchmarks" / "bench.py"
    if not script.exists():
        print(f"missing {script}", file=sys.stderr)
        return 1
    import runpy

    runpy.run_path(str(script), run_name="__main__")
    return 0


def _dashboard(port: int) -> int:
    try:
        import streamlit  # noqa: F401
    except ImportError:
        print("streamlit is not installed; install ptxlab with the viz extra", file=sys.stderr)
        return 1
    import subprocess

    app = Path(__file__).resolve().parents[2] / "viz" / "app.py"
    subprocess.run([sys.executable, "-m", "streamlit", "run", str(app), "--server.port", str(port)], check=False)
    return 0


def _load_module(path: Path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    if spec is None or spec.loader is None:
        raise SystemExit(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, "build_kernel"):
        raise SystemExit(f"{path} does not define build_kernel()")
    return module


def _pad(values: tuple[int, ...]) -> tuple[int, int, int]:
    dims = [1, 1, 1]
    for index, value in enumerate(values[:3]):
        dims[index] = int(value)
    return dims[0], dims[1], dims[2]


if __name__ == "__main__":
    raise SystemExit(main())
