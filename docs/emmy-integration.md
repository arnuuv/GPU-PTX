# Emmy integration notes

Inspected repository: [cloudrift-ai/emmy](https://github.com/cloudrift-ai/emmy) at `02cab261fdd89fd261441f2d02ddb400e94ead3a` (`02cab26`, 2026-10-13, "Refresh golden-bench kernel results and recover missing measurements"). PTXLab does not vendor that tree and does not modify it.

This note records how Emmy lowers kernels. PTXLab is not an Emmy backend.

## What the compiler actually lowers to

Emmy's own picture, from `emmy/compiler/ARCHITECTURE.md`, is three layers over one `Graph`:

```
PyTorch module
    trace/
Graph of frontend ops
    frontend decomposition and optimization
    loop lowering and fusion
Graph of LoopOp          (one LoopOp is one kernel)
    tile lift / cut / schedule
    lowering/kernel      TileOp to KernelOp
    lowering/cuda        KernelOp to CudaOp
backend/cuda             nvcc cubin, launched by the Rust runtime
```

`CudaOp` (`emmy/compiler/ir/cuda/ir.py`) stores `kernel_source` documented as a complete `__global__` function, plus `grid`, `block`, `smem_bytes`, `arg_order`, and optional TMA and symbolic-runtime metadata. `passes/lowering/cuda/010_lower_kernelop.py` renders each `KernelOp` body to that CUDA C string. The README's `--ir cuda` example is `extern "C" __global__` C++, with `__shared__`, `__syncthreads`, and `asm volatile(...)` for `cp.async`.

`backend/cuda/nvcc.py` compiles each kernel with offline `nvcc --cubin` into a content-addressed cache. The Rust runtime (`crates/emmy-runtime`) loads the cubin. A missing `nvcc` is an error. There is no NVRTC fallback in that path, and there is no step that writes a `.ptx` file as the program.

So the CUDA emitter is a CUDA C emitter. It is not a PTX emitter.

## Where PTX strings do appear

`emmy/compiler/ir/kernel/render.py` embeds PTX in the CUDA source as inline assembly: `cp.async`, `mbarrier.*`, `mma.sync.aligned`, `ldmatrix.sync.aligned`, and similar. Those sequences are operands of `asm volatile` inside a `__global__` function. `nvcc` parses them while compiling C++. They are not a PTX translation unit, they use instructions outside PTXLab's subset (`mma`, `ldmatrix`, bulk copy, mbarriers), and they cannot be fed to PTXLab's parser.

## Dialects that would matter to a backend

From `emmy/compiler/ir/ARCHITECTURE.md`:

| Dialect | Role |
| --- | --- |
| frontend / tensor | traced ops after decomposition (`ElementwiseOp`, `ReduceOp`, `MatmulOp`, layout ops) |
| loop | `LoopOp` plus `Load`, `Assign`, `Accum`, `Write`, `Loop` |
| tile | scheduled `TileOp` and `Fold` terms |
| kernel | thread-bound body: `Tile`, `Smem`, `Sync`, and the hardware statements the renderer lowers |
| cuda | `CudaOp` source and launch geometry |

Passes are legal rewrites. Ranking is evidence and the fitted prior, not a heuristic inside the pass (`pipeline/passes/ARCHITECTURE.md`). A PTX backend would have to preserve that split: it could lower a legal kernel, and it would not be the place that picks a tile because it looks faster.

Tests live under `tests/compiler`, mirroring the package, with architecture checks in `tests/architecture`. The README requires Python 3.12+ and, for running kernels, an NVIDIA GPU plus `nvcc`.

## Where a PTX backend could sit

The seam is after `KernelOp` exists and before `nvcc`, not by rewriting `CudaOp.kernel_source` and hoping it is PTX.

A future adapter would need to:

1. Recognise a KernelOp subset small enough for PTXLab: straight-line `f32`/`s32` arithmetic, predicated bounds, static shared memory, `bar.sync`, and the tiled GEMM pattern PTXLab already lowers. Emmy kernels also carry TMA descriptors, tensor-core fragments, async copies, symbolic `seq_len` extents, and multi-buffer outputs. PTXLab cannot lower those, and this repository does not try.
2. Lower that subset to PTXLab's AST or to the IR in `src/ptxlab/ir.py`, then call the existing emitter.
3. Live next to `backend/cuda` as a separate backend. The Rust runtime's cubin loader would stay on the nvcc path. A PTX file would still need `ptxas` or a driver JIT before a GPU launch. PTXLab's CPU simulator is not that launch path.
4. Keep launch geometry explicit. Emmy's `grid` / `block` entries are `GridDimSpec` values that may be symbolic expressions evaluated at launch. PTXLab launches are concrete integer triples.

`ptxlab.emmy_lower.lower_emmy_graph` does read Emmy objects, and only when the Emmy tree is on `PYTHONPATH`. It accepts one `ElementwiseOp` (`add`, `multiply`, `subtract`) on rank-1 static `f32` vectors, or one `ReduceOp` `sum` on axis 0 whose length is a power of two up to 1024. The tests in `tests/test_emmy_lower.py` were run against commit `02cab261` after `pip install frozendict`. They skip when `emmy` cannot be imported. `tanh` and every other op raise `EmmyLowerError`.

`ptxlab.emmy_adapter` is still a separate, local descriptor. It does not import Emmy. Its assumptions are the `ASSUMPTIONS` tuple, and tests check that the adapter source never imports Emmy.

## What was not verified

* `ElementwiseOp` and `ReduceOp` were constructed and lowered. `KernelOp`, `TileOp`, and `CudaOp` were not.
* Nothing here was checked against Emmy's cubin cache, golden benches, or runtime ABI.
* Agreement between a future extractor and `KernelOp.render` is an open question, not a result.
