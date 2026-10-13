# Simulation assumptions

There are two separate models. Mixing them up will misread the results.

## Functional interpreter

Implemented in `src/ptxlab/sim.py`. It defines the result of a kernel.

* Only the PTX subset in [ptx-subset.md](ptx-subset.md) runs. The parser rejects other instructions before execution.
* The host must be little-endian. Global memory is a set of byte allocations, one per pointer argument, 256-byte aligned. A 4-byte access must be aligned and must fall inside that allocation. Anything else raises `MemoryFault`.
* Pointer arguments are copied in at launch and copied back after the grid finishes. Scalar parameters are read with `ld.param` and are not addresses.
* Thread blocks run one after another, `gridDim.x` fastest. That order is a simulator choice. Do not treat it as a hardware schedule, and do not use it to justify a racy kernel.
* A warp is 32 consecutive linear ids: `tx + ty * blockDim.x + tz * blockDim.x * blockDim.y`. The last warp of a block may be short. Its launch mask is the live lanes only.
* The warp issues one instruction to its active mask. A predicated instruction writes a lane only when that lane's predicate is true. Inactive lanes do not touch memory.
* A predicated `bra` that some active lanes take and some do not splits the mask. Each side runs until the immediate post-dominator of that branch instruction, computed on the control-flow graph. The original mask then continues at the post-dominator. A uniform branch does not split. This is post-dominator reconvergence. It is not a model of a hardware reconvergence scoreboard.
* `bar.sync` suspends the warp. The block resumes when every warp that has not executed `ret` is waiting on the same barrier id. If the active mask is not the warp's launch mask, or if some warps return while others wait, the simulator raises `DivergentBarrierError`. Real hardware also treats a divergent barrier as undefined; PTXLab makes that a hard error.
* Shared memory is zero-filled, private to the block, and byte-addressed. It is not visible to other blocks.
* Registers start at 0. PTX leaves unread registers undefined. Zero is a simulator definition so that tests have a fixed starting state. Kernels still write a register before they read it.
* `u32` and `u64` wrap modulo 2^32 and 2^64. `s32` wraps modulo 2^32 and is then read as signed. `f32` add, sub, and mul are IEEE-754 binary32, round-to-nearest-even, matching the `.rn` modifier in the PTX. `shr.u32` uses the low 5 bits of the shift amount.
* There is no memory-consistency model beyond "lanes of one issued instruction run in lane order, and blocks run in order". Races are not detected and their results are not meaningful.

The interpreter is not an NVIDIA GPU and it will not execute arbitrary PTX from `nvcc` or from Emmy.

## Analytical timing model

Implemented in `src/ptxlab/perf.py`. It never changes functional results. Every cycle count it prints is an estimate.

Default machine, chosen as a small teaching device and not taken from a datasheet:

| Parameter | Default |
| --- | --- |
| SMs | 8 |
| Schedulers per SM | 4 |
| Threads / registers / shared bytes per SM | 2048 / 65536 / 65536 |
| Blocks per SM | 32 |
| Arithmetic issue | 1 cycle |
| First global 128-byte sector | 200 cycles |
| Each extra sector in that instruction | 32 cycles |
| Shared instruction | 20 cycles |
| Barrier | 20 cycles |

Further assumptions, also printed with each estimate:

* No cache, no scoreboard, and no latency hiding beyond dividing one wave's summed issue cost by the scheduler count.
* A global warp instruction costs 200 cycles if every participating lane hits the same 128-byte sector, plus 32 cycles per extra sector. Consecutive `f32` lanes coalesce. A stride of 32 elements does not. The 128-byte sector is a model choice, not a claim about a particular DRAM protocol.
* Declared registers, including predicates, all count toward the register file. Hardware packs predicates differently, so occupancy can be pessimistic.
* Occupancy is `min` of the thread, register, shared, and block caps, divided by 2048 threads. A kernel that fits zero blocks reports no cycle estimate.
* Blocks that fit concurrently form a wave. Further waves are serial. The functional simulator still runs every block one at a time; that schedule is not the timing model.
* Utilization is issued warp instructions divided by `cycles * SMs * schedulers`.

`ptxlab bench` prints these estimates next to the functional error. It does not print a speedup, and it must not be quoted as one.

## GPU toolchain

If `ptxas` is on `PATH`, `tests/test_cuda_optional.py` asks it to assemble `vector_add` for `sm_70`. On this machine neither `ptxas` nor `nvcc` was available, so that test skips. A skip is not a successful GPU run.
