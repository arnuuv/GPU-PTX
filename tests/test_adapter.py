"""The Emmy adapter stays on PTXLab's own kernel type."""

from pathlib import Path

import numpy as np

import pytest

from ptxlab.emmy_adapter import ASSUMPTIONS, EmmyKernelDescriptor, EmmyPtxAdapter
from ptxlab.errors import LaunchError
from ptxlab.kernels import vector_add
from ptxlab.parse import parse


def test_adapter_emits_ptx_and_matches_numpy() -> None:
    adapter = EmmyPtxAdapter()
    kernel = vector_add()
    descriptor = EmmyKernelDescriptor("vector_add", kernel, (1, 1, 1), (32, 1, 1))
    program = parse(adapter.emit_ptx(descriptor))
    assert program.name == "vector_add"
    assert program.address_size == 64
    a = np.arange(32, dtype=np.float32)
    b = np.ones(32, np.float32)
    c = np.zeros(32, np.float32)
    adapter.simulate(descriptor, {"a": a, "b": b, "c": c, "n": 32})
    np.testing.assert_array_equal(c, a + b)
    assert any("not PTX" in line or "CUDA C" in line for line in ASSUMPTIONS)


def test_adapter_rejects_a_block_the_kernel_fixed() -> None:
    from ptxlab.kernels import reduction

    adapter = EmmyPtxAdapter()
    descriptor = EmmyKernelDescriptor("reduce", reduction(32), (1, 1, 1), (64, 1, 1))
    with pytest.raises(LaunchError, match="expects block"):
        adapter.emit_ptx(descriptor)


def test_adapter_source_does_not_import_emmy() -> None:
    text = Path("src/ptxlab/emmy_adapter.py").read_text(encoding="utf-8")
    assert "import emmy" not in text
    assert "from emmy" not in text
