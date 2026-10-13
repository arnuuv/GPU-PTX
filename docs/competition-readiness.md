# Competition readiness

Audit of PTXLab before the October 2026 compiler upgrade. The RiftStack Emmy Hackathon runs 21–27 October 2026. Its problems and judging criteria were not published when this file was written. Nothing here predicts a winning submission.

Baseline before that upgrade: `51 passed, 1 skipped` (`tests/test_cuda_optional.py` skips because `ptxas` is not installed). Host: macOS, no `nvcc`, no `ptxas`, no `libcuda`.

Emmy was inspected at `02cab261fdd89fd261441f2d02ddb400e94ead3a`. See [emmy-integration.md](emmy-integration.md).

## Verified on the CPU

- Typed AST to three-address IR, validator, and text dump.
- PTX emission for `.version 7.0`, `.target sm_70`, `.address_size 64`, with a checked target table.
- Parser that accepts that subset and rejects other opcodes.
- Functional interpreter: warps of 32, post-dominator reconvergence, global and shared memory, `bar.sync`.
- Kernels with exact `f32` agreement against their references on the tested shapes: vector add, vector multiply, SAXPY, power-of-two block reduction, tiled GEMM.
- Bounds checks that avoid out-of-range accesses, and an unmasked access that faults.
- Analytical cycle, transaction, and occupancy model. Those numbers are estimates.

## Not verified

- No PTX file has been assembled with `ptxas`.
- No kernel has been launched on an NVIDIA GPU.
- No CUDA driver timing, register-usage report, or Compute Sanitizer run exists.
- The Emmy adapter does not read Emmy IR. It wraps PTXLab's own kernel type.
- There is no CI workflow.

## Unsupported or incomplete

- One virtual register becomes one physical register. No folding, DCE, CSE, or scheduling.
- No `fma`, `div`, `sqrt`, `ex2`, atomics, vector loads, half precision, warp shuffle, `mma`, `ldmatrix`, or `cp.async`.
- No layer norm, RMSNorm, or softmax.
- Autotuning is absent. Simulator cycles must not be reported as speedups.
- BF16, tensor-core MMA, and asynchronous copies are out of the subset.
- The interpreter runs blocks one at a time and is not a performance model of a GPU.

This file is updated again after the upgrade so the lists above are not left as the final claim.
