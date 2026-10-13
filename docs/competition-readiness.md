# Competition readiness

Updated after the compiler upgrade. The RiftStack Emmy Hackathon is 21–27 October 2026. Its problems and judging criteria were not published. This tree does not implement an unreleased task.

Emmy was executed, for the narrow lowerer only, at `02cab261fdd89fd261441f2d02ddb400e94ead3a`. See [emmy-integration.md](emmy-integration.md).

## Verified on this machine

`pytest` without Emmy on `PYTHONPATH`: 70 passed, 2 skipped. The skips are `ptxas` (not installed) and the Emmy lowerer (the package is not installed). With `PYTHONPATH` pointed at that Emmy checkout, `tests/test_emmy_lower.py` passed: an `ElementwiseOp("add")` graph and a length-32 `ReduceOp("sum")` matched the PTXLab interpreter.

Also verified:

- IR validation, lowering, greedy register allocation, and the optimisation passes in `src/ptxlab/opt.py`.
- Constant folding, integer strength reduction (`x * 2` to `x + x`), CSE of the address scale, copy propagation, DCE, LICM, optional unrolling, and list scheduling. Equivalence is against the interpreter. No pass has a measured GPU speedup.
- Vector add, multiply, SAXPY, `fma.rn.f32` SAXPY, f16 add via f32, aligned `v4` add, integer `atom.add.u32`, block reduction, tiled GEMM, layer norm, RMSNorm, and softmax.
- Static unit-stride and strided classifications in `ptxlab.analysis`.
- CUDA driver detection. `libcuda` is absent, so `ptxlab validate --backend cuda` exits 2 and records that no GPU measurement was collected.
- Analytical cycle estimates remain labelled as estimates. `ptxlab report` writes interpreter, analytical, and GPU sections; the GPU latency is null here.

## Not verified

- No kernel was assembled by `ptxas`.
- No kernel was launched on an NVIDIA GPU.
- No Compute Sanitizer run.
- The Docker image in `docker/Dockerfile` was not built.
- Autotune did not rank candidates, because ranking requires device timings.
- `ex2.approx.ftz.f32` was not compared with a GPU special-function unit. The interpreter uses flushed `exp2`.

## Still unsupported

- Warp shuffle, BF16, `mma`, `ldmatrix`, `cp.async`, and `ld.global.v2`.
- Emmy graphs other than the two patterns above. No `KernelOp`, TMA, or symbolic shape is lowered.
- A median GPU benchmark table. The CUDA path times `cuLaunchKernel` plus `cuCtxSynchronize` only when a device exists.

## How to run the missing device checks

Follow [cuda-runner.md](cuda-runner.md). The optional GitHub workflow `.github/workflows/gpu.yml` is `workflow_dispatch` and expects a self-hosted runner labelled `nvidia`.
