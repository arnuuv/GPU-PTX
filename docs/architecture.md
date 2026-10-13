# Architecture

PTXLab compiles a small kernel language to a documented NVIDIA PTX subset and executes that subset on the CPU. Timing estimates are a separate analytical model.

```
KernelBuilder AST
    lower()        typed three-address IR
    validate()     types, symbols, targets, reachability
    emit()         PTX text
    parse()        the same subset, or UnsupportedInstruction
    simulate()     functional interpreter
    estimate()     optional cycle / occupancy model
```

## AST

`ptxlab.dsl.KernelBuilder` builds a structured kernel: pointer and scalar parameters, static shared arrays, assignments, predicated `when` regions, counted `for_range` loops, and barriers. Element types at the language boundary are `f32`, `s32`, and `u32`. Index arithmetic is `u32`. Shapes name scalar parameters or static extents.

Expressions are not Python booleans. `when()` is the only branch.

## IR

`KernelIR` is a list of basic blocks. Each instruction writes at most one virtual register. Terminators are `br`, `cbr`, and `ret`. The validator checks:

- PTX ISA version, target, and 64-bit address size against `toolchain.py`
- parameter and shared-memory declarations
- a single type per virtual register
- definitions before uses in block-list order
- branch targets and reachability from `entry`

The dump (`dump_ir`) is the text shown in the viewer. Virtual registers stay named there. Physical PTX names are assigned only by the emitter.

## PTX

`emit.py` maps each IR operation to one PTX mnemonic. `f32` arithmetic uses `.rn`. Global addresses are `u64`, formed with `mul.wide.u32` of the element index by 4 and `add.u64` of the parameter base. Bounds checks are `setp` plus a predicated `bra`. The parser accepts that shape and rejects other opcodes.

## Simulator

`sim.py` is the functional definition. Blocks run in order. Warps are 32 consecutive linear thread ids. A divergent predicated branch splits the active mask until the immediate post-dominator, then the mask joins. `bar.sync` waits until every warp that has not returned is at the same barrier; a split mask at a barrier is an error. Details and the list of things this does not model are in [simulation-assumptions.md](simulation-assumptions.md).

## Performance model

`perf.py` reads the functional trace. It does not change results. Occupancy is a limit calculation from threads, declared registers, and shared bytes. Cycle estimates add configured issue and memory costs, then spread a wave across the configured schedulers. The report object is marked `analytical estimate`.

## What is intentionally absent

- A general PTX assembler or a claim to run arbitrary CUDA
- Shared-memory atomics, warp shuffles, tensor-core instructions, TMA, or `cp.async`
- Device-wide reduction beyond one partial sum per block
- Measured GPU times. `ptxas` / `nvcc` are used only when they are on `PATH`, and the test skips otherwise
- An Emmy `Graph` or `KernelOp` importer. See [emmy-integration.md](emmy-integration.md)
