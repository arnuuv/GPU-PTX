# PTXLab

PTXLab is a small, tested compiler from a typed kernel language to a documented subset of NVIDIA PTX, plus a CPU interpreter for exactly the instructions that compiler emits. An optional analytical model estimates cycles and occupancy. Those estimates are not GPU measurements and are not speedups.

Emmy's CUDA backend emits `__global__` CUDA C and cubins, not PTX. See [docs/emmy-integration.md](docs/emmy-integration.md).

References: [Emmy](https://github.com/cloudrift-ai/emmy), [PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html), [CUDA C programming guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html).

## Setup

Python 3.12 or newer. A GPU is not required.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

The viewer needs an extra: `pip install -e ".[dev,viz]"`.

## Commands

```bash
ptxlab compile examples/vector_add.py
ptxlab simulate examples/vector_add.ptx
ptxlab validate examples/vector_add.py --backend cpu
ptxlab run examples/vector_add.py --backend cuda
ptxlab test
ptxlab bench
ptxlab bench --backend cuda
ptxlab autotune
ptxlab report
ptxlab dashboard
```

`validate` and `run` accept `--backend cpu` or `--backend cuda`. On a machine without `libcuda`, the CUDA backend exits 2 and does not print interpreter numbers as device timings. See [docs/cuda-runner.md](docs/cuda-runner.md).

`autotune` checks vector-add block sizes for exact agreement with NumPy. It ranks them only when a GPU measurement exists. `report` writes `benchmarks/results/report.json` with separate interpreter, analytical, and GPU sections.

`compile` writes `examples/vector_add.ptx` and a `.launch.json` built from the example's `sample()`. `simulate` runs that launch on the CPU interpreter and prints the functional error against the reference stored in the launch file.

`vector_add` for 128 `f32` values, block 64, on the analytical model (8 SMs, the defaults in `ptxlab.perf`):

```
kernel vector_add
instructions 108  divergent_branches 0
global_transactions 12  occupancy 1.000
estimated_cycles 9984  (analytical estimate, not a GPU measurement)
c max_abs_error 0.000e+00
reference match: exact on floating outputs
```

`ptxlab bench` prints the same kind of line for vector add, vector multiply, and SAXPY at n = 256. A run on this machine:

```
kernel                  n    max_abs_err   transactions   est_cycles  occupancy
vector_add            256      0.000e+00             24         9984      1.000
vector_mul            256      0.000e+00             24         9984      1.000
saxpy                 256      0.000e+00             24        10016      1.000
```

`est_cycles` is the model in [docs/simulation-assumptions.md](docs/simulation-assumptions.md). Re-run the command rather than treating the table as a hardware result. `max_abs_err` is the functional comparison with NumPy and is exact for these kernels.

The generated vector-add entry starts:

```ptx
.version 7.0
.target sm_70
.address_size 64
.visible .entry vector_add (
    .param .u64 a,
    .param .u64 b,
    .param .u64 c,
    .param .u32 n
)
```

The index is `%ctaid.x * %ntid.x + %tid.x`, compared with `setp.lt.u32`, and the in-bounds path computes a 64-bit address with `mul.wide.u32` and `add.u64` before `ld.global.f32` / `st.global.f32`. The full file is `examples/vector_add.ptx`.

## Kernels

| Example | What it does |
| --- | --- |
| `examples/vector_add.py` | `c[i] = a[i] + b[i]` with a bounds check |
| `examples/vector_mul.py` | elementwise multiply |
| `examples/saxpy.py` | `z[i] = alpha * x[i] + y[i]` |
| `examples/reduction.py` | one partial sum per block, shared-memory tree, `bar.sync` |
| `examples/matmul.py` | 8×8 tiled GEMM, static shared tiles, two barriers per K tile |

Reduction and matmul fix the block size in the PTXLab block comment. Launching a different block is an error. Partial sums are the per-block results; adding them on the host is a separate step, and `tests/test_kernels.py` checks both the tree and a looser comparison with `numpy.sum`.

## Dashboard

`ptxlab dashboard` opens a Streamlit page with a kernel selector, the IR dump, the PTX, instruction counts, a short warp trace, the memory-access timeline, occupancy, the analytical cycle estimate, the max absolute error against the reference, and the model assumptions. Cycle numbers on that page are labelled as estimates.

## Tests

`pytest` covers IR validation, lowering, PTX round-trip, unsupported opcodes, elementwise and integer kernels, divergent branches, bounds faults, barriers, reduction, tiled matmul, the timing model, the CLI, and the adapter. Randomized lengths and block sizes are in `tests/test_sim.py` and `tests/test_kernels.py`. Every supported opcode used by the kernels is executed in `tests/test_cli.py`.

`tests/test_cuda_optional.py` calls `ptxas -arch=sm_70` when `ptxas` is installed. It skips otherwise. A skip is not a GPU result.

## Layout

```
src/ptxlab/     compiler, parser, simulator, model, CLI
examples/       kernels, emitted PTX, launch records
tests/          pytest
benchmarks/     analytical benchmark script
viz/app.py      Streamlit shell
docs/           architecture, PTX subset, simulator, Emmy notes
```

## Limits

* The simulator executes this subset only. It will not run general CUDA or Emmy's inline PTX.
* f16 values are loaded and stored, then added in f32. BF16, warp shuffle, tensor cores, TMA, and `cp.async` are absent.
* `ex2.approx.ftz.f32` is flushed `exp2` in the interpreter, not a GPU special-function result.
* Shared memory and barriers are block-local. A divergent barrier is an error.
* Reduction block sizes are compile-time powers of two. The matmul tile is a compile-time square, at most 32. Layer norm, RMSNorm, and softmax use one power-of-two block per row.
* Register occupancy counts predicate registers as full registers.
* No measured device timeline. `nvcc` and `ptxas` were not on `PATH` when this tree was built.
* Emmy lowering covers one elementwise pattern and one power-of-two sum. It is not a KernelOp backend.

More detail: [docs/architecture.md](docs/architecture.md), [docs/ptx-subset.md](docs/ptx-subset.md), [docs/simulation-assumptions.md](docs/simulation-assumptions.md), [docs/emmy-integration.md](docs/emmy-integration.md).
