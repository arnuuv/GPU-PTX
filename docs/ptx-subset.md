# PTX subset

Default module header, unless a kernel overrides it:

```
.version 7.0
.target sm_70
.address_size 64
```

`toolchain.py` accepts ISA 7.0 through 8.5 and these targets, each with the minimum ISA that introduced it in NVIDIA's PTX ISA specification: `sm_70` (6.0), `sm_72` (6.1), `sm_75` (6.3), `sm_80` (7.0), `sm_86` (7.1), `sm_87` (7.1), `sm_89` (7.8), `sm_90` (7.8). A target newer than the chosen version is rejected. Address size 32 is rejected because every global pointer this compiler emits is 64-bit.

The instruction set is exactly the opcodes in `parse.ALLOWED_OPS`. The emitter does not produce anything else. The parser raises `UnsupportedInstruction` for any other opcode, including vector loads such as `ld.global.v4.f32`.

## IR operation to PTX

| IR | PTX |
| --- | --- |
| `const`, `mov` | `mov.u32`, `mov.s32`, `mov.u64`, `mov.f32`, or `mov.pred`. `f32` immediates are `0f` hex |
| `tid` / `ctaid` / `ntid` | `mov.u32 rd, %tid.{x,y,z}` and the matching `%ctaid` / `%ntid` |
| `add` / `sub` | `add.rn.f32`, `sub.rn.f32`, or `add`/`sub` with `.u32`, `.s32`. `add.u64` is used for addresses |
| `mul` | `mul.rn.f32` or `mul.lo.{u32,s32}` |
| `mul_wide` | `mul.wide.u32` producing a `.u64` byte offset |
| `mad_lo` | `mad.lo.{u32,s32}` |
| `shr` | `shr.u32` |
| `cmp` | `setp.{eq,ne,lt,le,gt,ge}.{u32,s32}` |
| `and_pred` / `or_pred` / `not_pred` | `and.pred`, `or.pred`, `not.pred` |
| `ld_param` | `ld.param.{u64,u32,s32,f32}` |
| `ld_global` / `st_global` | `ld.global` / `st.global` of `.f32`, `.s32`, `.u32` through a `.u64` address |
| `shared_base` | `mov.u32 rd, symbol` |
| `ld_shared` / `st_shared` | `ld.shared` / `st.shared` through a `.u32` byte address |
| `bar` | `bar.sync imm` with an id in 0..15 |
| `cbr` | `@p bra then` followed by `bra else` |
| `br` / `ret` | `bra`, `ret` |

`.rn` is required on `f32` `add` and `mul` in the ISA versions this project targets. Integer multiplies that keep a 32-bit result use `.lo`. The 64-bit product used for addressing is `mul.wide.u32`, which is defined for a `.u32` times a `.u32` into a `.u64`.

## Parameters and shared memory

Pointer parameters are `.param .u64`. Scalar parameters are `.param .u32`, `.s32`, or `.f32`. Shared declarations are `.shared .align 4 .f32 name[N]` (or `.s32` / `.u32`) with a static length. A kernel that needs a fixed block, such as reduction or tiled GEMM, records it in a comment:

```
// .ptxlab_block 32 1 1
```

That comment is PTXLab metadata. It is not a PTX directive. The simulator checks the launch block against it.

## Addressing

For a 4-byte element at index `i` and a 64-bit base `p`:

```
mul.wide.u32 %rd_off, %r_i, 4;
add.u64 %rd_addr, %rd_base, %rd_off;
ld.global.f32 %f, [%rd_addr];
```

Row-major index linearisation (`i0 * stride + i1`) happens in `u32` before the widening multiply. The bounds predicate is tested before the load or store, so out-of-range threads take the other branch and do not issue the memory operation.

## Not in the subset

`ld.global.v2` / `v4`, `atom`, `shfl`, `mma`, `cp.async`, `ldmatrix`, texture operations, predicates on `bar.sync`, and 16-bit floats. Emmy's CUDA renderer uses several of those as inline `asm` inside C++; they are not accepted here.
