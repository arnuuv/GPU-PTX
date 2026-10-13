"""CUDA detection and parameter packing. No device is required."""

import numpy as np

from ptxlab.cli import main
from ptxlab.cuda_host import detect, pack_scalar


def test_detect_reports_availability_without_fabricating_a_device() -> None:
    status = detect()
    if status.available:
        assert status.device_count >= 1
        assert status.device_name
        assert status.kind == "nvidia-driver"
    else:
        assert status.device_count == 0
        assert "libcuda" in status.reason or "driver" in status.reason.lower() or "zero devices" in status.reason
        assert status.kind == "unavailable"


def test_scalar_packing_uses_the_parameter_width() -> None:
    assert pack_scalar("u32", 32).value == 32
    assert pack_scalar("s32", -2).value == -2
    packed = pack_scalar("f32", 1.5)
    assert np.float32(packed.value) == np.float32(1.5)
    assert pack_scalar("u64", 0x1000).value == 0x1000


def test_cpu_validate_matches_and_cuda_does_not_pretend(tmp_path) -> None:
    cpu = main(["validate", "examples/vector_add.py", "--backend", "cpu"])
    assert cpu == 0
    cuda = main(["validate", "examples/vector_add.py", "--backend", "cuda"])
    status = detect()
    if status.available:
        assert cuda == 0
    else:
        assert cuda == 2
