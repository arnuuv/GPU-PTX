"""PTX ISA version and target checks.

The table records the minimum PTX ISA version that introduced each target in
NVIDIA's public PTX ISA specification. It is a compatibility check for the
directives this compiler emits, not a query of a local ``ptxas``.
"""

from __future__ import annotations

from ptxlab.errors import ToolchainError

# (major, minor) -> minimum ISA that introduced the architecture.
TARGET_MIN_ISA: dict[str, tuple[int, int]] = {
    "sm_70": (6, 0),
    "sm_72": (6, 1),
    "sm_75": (6, 3),
    "sm_80": (7, 0),
    "sm_86": (7, 1),
    "sm_87": (7, 1),
    "sm_89": (7, 8),
    "sm_90": (7, 8),
}

# Versions this emitter and parser know. The instruction subset itself is
# older than 7.0; versions outside this list are rejected rather than guessed.
KNOWN_ISA: tuple[tuple[int, int], ...] = tuple(
    (major, minor)
    for major, minors in (
        (7, range(0, 9)),
        (8, range(0, 6)),
    )
    for minor in minors
)

DEFAULT_ISA = (7, 0)
DEFAULT_TARGET = "sm_70"
DEFAULT_ADDRESS_SIZE = 64


def parse_version(text: str) -> tuple[int, int]:
    parts = text.split(".")
    if len(parts) != 2:
        raise ToolchainError(f"PTX version must be major.minor, got {text!r}")
    try:
        version = (int(parts[0]), int(parts[1]))
    except ValueError as exc:
        raise ToolchainError(f"PTX version must be major.minor, got {text!r}") from exc
    return version


def format_version(version: tuple[int, int]) -> str:
    return f"{version[0]}.{version[1]}"


def validate_toolchain(version: tuple[int, int], target: str, address_size: int) -> None:
    if address_size != 64:
        raise ToolchainError(
            f"address size {address_size} is not supported; PTXLab emits 64-bit global addresses"
        )
    if version not in KNOWN_ISA:
        known = ", ".join(format_version(item) for item in KNOWN_ISA)
        raise ToolchainError(
            f"PTX ISA {format_version(version)} is outside the versions PTXLab can target ({known})"
        )
    if target not in TARGET_MIN_ISA:
        known_targets = ", ".join(sorted(TARGET_MIN_ISA))
        raise ToolchainError(f"target {target} is not in the supported set ({known_targets})")
    minimum = TARGET_MIN_ISA[target]
    if version < minimum:
        raise ToolchainError(
            f"target {target} requires PTX ISA {format_version(minimum)} or newer, got {format_version(version)}"
        )
