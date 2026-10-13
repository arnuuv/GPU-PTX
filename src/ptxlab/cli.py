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
    compile_p.add_argument("--no-opt", action="store_true")

    simulate_p = sub.add_parser("simulate", help="run an emitted PTX file on the CPU simulator")
    simulate_p.add_argument("ptx", type=Path)
    simulate_p.add_argument("--launch", type=Path, help="launch JSON; defaults to the .launch.json beside the PTX")

    run_p = sub.add_parser("run", help="compile an example and execute it")
    run_p.add_argument("source", type=Path)
    run_p.add_argument("--backend", choices=("cpu", "cuda"), default="cpu")
    run_p.add_argument("--arch", default=None, help="ptxas -arch value; default is the PTX .target")

    val_p = sub.add_parser("validate", help="compare an example with its reference")
    val_p.add_argument("source", type=Path)
    val_p.add_argument("--backend", choices=("cpu", "cuda"), default="cpu")

    sub.add_parser("test", help="run the pytest suite")
    bench_p = sub.add_parser("bench", help="run benchmarks")
    bench_p.add_argument("--backend", choices=("cpu", "cuda"), default="cpu")
    dash = sub.add_parser("dashboard", help="open the local Streamlit viewer")
    dash.add_argument("--port", type=int, default=8501)

    args = parser.parse_args(argv)
    if args.command == "compile":
        return _compile(args.source, args.output, no_opt=args.no_opt)
    if args.command == "simulate":
        return _simulate(args.ptx, args.launch)
    if args.command == "run":
        return _execute_example(args.source, args.backend, validate=False)
    if args.command == "validate":
        return _execute_example(args.source, args.backend, validate=True)
    if args.command == "test":
        return _test()
    if args.command == "bench":
        return _bench(args.backend)
    if args.command == "dashboard":
        return _dashboard(args.port)
    parser.error(f"unknown command {args.command}")
    return 2


def _compile(source: Path, output: Path | None, *, no_opt: bool = False) -> int:
    from ptxlab.opt import OptimizeConfig

    module = _load_module(source)
    kernel = module.build_kernel()
    text = compile_kernel(kernel, config=OptimizeConfig.none() if no_opt else None)
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


def _execute_example(source: Path, backend: str, *, validate: bool) -> int:
    from ptxlab.cuda_host import detect, launch
    from ptxlab.errors import CudaUnavailable

    module = _load_module(source)
    if not hasattr(module, "sample"):
        print(f"{source} does not define sample()", file=sys.stderr)
        return 1
    grid, block, args, compare = module.sample()
    grid3, block3 = _pad(tuple(grid)), _pad(tuple(block))
    kernel = module.build_kernel()
    ptx = compile_kernel(kernel)
    program = parse(ptx)
    print(f"backend {backend}")
    if backend == "cpu":
        result = simulate(program, grid=grid3, block=block3, args=args)
        print(f"interpreter_instructions {result.instructions}")
        print("result_source cpu-interpreter")
    else:
        status = detect()
        print(status.reason)
        if not status.available:
            print("result_source unavailable")
            print("No NVIDIA GPU measurement was collected.")
            return 2
        try:
            elapsed_ms = launch(program, ptx, grid=grid3, block=block3, args=args)
        except CudaUnavailable as exc:
            print(exc, file=sys.stderr)
            return 2
        print(f"device {status.device_name}")
        print(f"launch_sync_ms {elapsed_ms:.4f}")
        print("result_source nvidia-gpu")
    return _compare(args, compare, strict=validate)


def _compare(args: dict, compare: dict, *, strict: bool) -> int:
    worst = 0.0
    for name, expected in compare.items():
        actual = args[name]
        if not isinstance(actual, np.ndarray):
            continue
        if actual.dtype == np.float32:
            error = float(np.max(np.abs(actual.astype(np.float64) - expected.astype(np.float64))))
            worst = max(worst, error)
            print(f"{name} max_abs_error {error:.3e}")
            if strict and error != 0.0:
                return 1
        else:
            mismatches = int(np.count_nonzero(actual != expected))
            print(f"{name} mismatches {mismatches}")
            if mismatches:
                return 1
    if compare and worst == 0.0:
        print("reference match: exact on floating outputs")
    return 0


def _bench(backend: str) -> int:
    if backend == "cuda":
        from ptxlab.cuda_host import detect

        status = detect()
        print(status.reason)
        if not status.available:
            print("GPU benchmarks were not collected. Interpreter timings are not a substitute.")
            return 2
        print(f"device {status.device_name} driver {status.driver_version}")
        root = Path(__file__).resolve().parents[2] / "examples"
        for name in ("vector_add.py", "saxpy.py", "vector_mul.py"):
            code = _execute_example(root / name, "cuda", validate=True)
            if code != 0:
                return code
        print("timings are host-side launch plus cuCtxSynchronize, including driver overhead")
        print("they are single launches, not a warmed median")
        return 0
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
