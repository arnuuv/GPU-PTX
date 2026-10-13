"""Analytical timing model.

Every number this module returns is an estimate under the assumptions on
``ArchConfig``. It is not a measurement from an NVIDIA GPU, ``ptxas``, or a
profiler. Functional results come from :mod:`ptxlab.sim` and do not depend on
this model.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil

from ptxlab.parse import Program
from ptxlab.sim import SimResult


@dataclass(frozen=True)
class ArchConfig:
    """Configurable machine description used only by the analytical model.

    Defaults are round teaching numbers (a small device with 8 SMs, 2048
    threads and 64K registers per SM). They are not the datasheet of a
    particular GPU.
    """

    num_sms: int = 8
    schedulers_per_sm: int = 4
    max_threads_per_sm: int = 2048
    max_regs_per_sm: int = 65536
    max_shared_bytes_per_sm: int = 65536
    max_blocks_per_sm: int = 32
    arith_issue_cycles: int = 1
    global_latency_cycles: int = 200
    shared_latency_cycles: int = 20
    barrier_latency_cycles: int = 20
    extra_sector_cycles: int = 32
    sector_bytes: int = 128


@dataclass(frozen=True)
class PerfEstimate:
    """Model output. ``estimated_cycles`` is an analytical estimate."""

    estimated_cycles: int | None
    memory_transactions: int
    shared_instructions: int
    occupancy: float
    blocks_per_sm: int
    waves: int | None
    utilization: float | None
    divergent_branches: int
    assumptions: tuple[str, ...]

    @property
    def kind(self) -> str:
        return "analytical estimate"


def occupancy(
    *,
    threads_per_block: int,
    regs_per_thread: int,
    shared_bytes: int,
    arch: ArchConfig | None = None,
) -> tuple[int, float]:
    machine = arch or ArchConfig()
    if threads_per_block <= 0:
        return 0, 0.0
    by_threads = machine.max_threads_per_sm // threads_per_block
    reg_footprint = regs_per_thread * threads_per_block
    by_regs = machine.max_regs_per_sm // reg_footprint if reg_footprint else machine.max_blocks_per_sm
    by_shared = (
        machine.max_shared_bytes_per_sm // shared_bytes if shared_bytes else machine.max_blocks_per_sm
    )
    blocks = max(0, min(machine.max_blocks_per_sm, by_threads, by_regs, by_shared))
    if blocks == 0:
        return 0, 0.0
    return blocks, (blocks * threads_per_block) / machine.max_threads_per_sm


def memory_transactions(result: SimResult, *, sector_bytes: int = 128) -> int:
    """Count 128-byte sectors touched by each global warp instruction.

    One sector is one modelled transaction. Consecutive 4-byte lanes that stay
    inside a sector coalesce; a lane that steps into another sector adds one.
    This is not the sector size of a specific DRAM protocol.
    """

    total = 0
    for access in result.mem_accesses:
        if access.op.startswith("ld.global") or access.op.startswith("st.global"):
            total += len({addr // sector_bytes for addr in access.addrs})
    return total


def estimate(
    program: Program,
    result: SimResult,
    *,
    grid: tuple[int, int, int],
    block: tuple[int, int, int],
    arch: ArchConfig | None = None,
) -> PerfEstimate:
    machine = arch or ArchConfig()
    threads = block[0] * block[1] * block[2]
    regs = sum(program.reg_counts.values())
    shared_bytes = sum(item.nbytes for item in program.shared)
    blocks_sm, occ = occupancy(
        threads_per_block=threads,
        regs_per_thread=regs,
        shared_bytes=shared_bytes,
        arch=machine,
    )
    transactions = memory_transactions(result, sector_bytes=machine.sector_bytes)
    shared_ops = sum(
        count for op, count in result.inst_counts.items() if op.startswith("ld.shared") or op.startswith("st.shared")
    )
    barriers = result.inst_counts.get("bar.sync", 0)
    global_ops = sum(
        1
        for access in result.mem_accesses
        if access.op.startswith("ld.global") or access.op.startswith("st.global")
    )
    other = result.instructions - shared_ops - barriers - global_ops
    global_cost = 0
    for access in result.mem_accesses:
        if access.op.startswith("ld.global") or access.op.startswith("st.global"):
            sectors = len({addr // machine.sector_bytes for addr in access.addrs})
            global_cost += machine.global_latency_cycles + max(0, sectors - 1) * machine.extra_sector_cycles
    issue_cost = (
        other * machine.arith_issue_cycles
        + shared_ops * machine.shared_latency_cycles
        + barriers * machine.barrier_latency_cycles
        + global_cost
    )
    total_blocks = grid[0] * grid[1] * grid[2]
    assumptions = _assumptions(machine, regs)
    if blocks_sm == 0 or total_blocks == 0:
        return PerfEstimate(
            None,
            transactions,
            shared_ops,
            occ,
            blocks_sm,
            None,
            None,
            result.divergent_branches,
            assumptions,
        )
    slots = blocks_sm * machine.num_sms
    waves = ceil(total_blocks / slots)
    # Summed issue cost is spread across the schedulers of one SM, then across
    # the blocks that share that SM in a wave. Waves themselves are serial.
    sm_wave_cycles = ceil(issue_cost * blocks_sm / max(1, total_blocks) / machine.schedulers_per_sm)
    cycles = sm_wave_cycles * waves
    capacity = cycles * machine.num_sms * machine.schedulers_per_sm
    utilization = (result.instructions / capacity) if capacity else None
    return PerfEstimate(
        cycles,
        transactions,
        shared_ops,
        occ,
        blocks_sm,
        waves,
        utilization,
        result.divergent_branches,
        assumptions,
    )


def _assumptions(arch: ArchConfig, regs: int) -> tuple[str, ...]:
    return (
        "Analytical estimate, not a measured GPU time and not a speedup.",
        f"{arch.num_sms} SMs, {arch.schedulers_per_sm} schedulers each, "
        f"{arch.max_threads_per_sm} threads, {arch.max_regs_per_sm} registers, "
        f"{arch.max_shared_bytes_per_sm} shared bytes per SM.",
        "No cache, no scoreboard, and no latency hiding beyond dividing a wave's "
        "summed issue cost by the scheduler count.",
        f"Global accesses cost {arch.global_latency_cycles} cycles for the first "
        f"{arch.sector_bytes}-byte sector and {arch.extra_sector_cycles} cycles "
        "for each extra sector. Shared and barrier costs are flat.",
        f"Register count {regs} includes predicate registers, which real hardware packs separately.",
        "Blocks in a wave share an SM up to the occupancy limit. Further waves are serial.",
        "Functional simulation still runs blocks one at a time; that order is not this timing model.",
    )
