"""Project error types."""


class PtxLabError(Exception):
    """Base class for PTXLab failures."""


class IrError(PtxLabError):
    """Typed IR failed validation."""


class ToolchainError(PtxLabError):
    """PTX version, target, or address size is not in the supported set."""


class LoweringError(PtxLabError):
    """AST could not be lowered."""


class EmitError(PtxLabError):
    """IR could not be rendered as PTX."""


class ParseError(PtxLabError):
    """Text is not in the PTX subset this project parses."""


class UnsupportedInstruction(PtxLabError):
    """An instruction is outside the implemented PTX subset."""


class MemoryFault(PtxLabError):
    """A simulated access missed an allocation or was misaligned."""


class LaunchError(PtxLabError):
    """Grid, block, or kernel arguments do not match the kernel."""


class DivergentBarrierError(PtxLabError):
    """A barrier was reached with a split mask or a missing warp."""


class SimulationError(PtxLabError):
    """Functional simulation could not continue."""


class CudaUnavailable(PtxLabError):
    """The NVIDIA driver or a GPU is not available on this host."""


class CudaLaunchError(PtxLabError):
    """The CUDA driver rejected a real-device launch."""
