"""Optional NVIDIA toolchain check. Skips when ptxas is absent."""

import shutil
import subprocess

import pytest

from ptxlab.driver import compile_kernel
from ptxlab.kernels import vector_add


def toolchain_note() -> str:
    ptxas = shutil.which("ptxas")
    nvcc = shutil.which("nvcc")
    if ptxas is None and nvcc is None:
        return "ptxas and nvcc are not on PATH; GPU validation was skipped"
    return f"ptxas={ptxas} nvcc={nvcc}"


def test_missing_toolchain_is_reported() -> None:
    note = toolchain_note()
    assert "skipped" in note or "ptxas=" in note


@pytest.mark.skipif(shutil.which("ptxas") is None, reason="ptxas is not installed")
def test_ptxas_accepts_vector_add(tmp_path) -> None:
    source = tmp_path / "vector_add.ptx"
    source.write_text(compile_kernel(vector_add()), encoding="utf-8")
    compiled = tmp_path / "vector_add.cubin"
    result = subprocess.run(
        ["ptxas", "-arch=sm_70", str(source), "-o", str(compiled)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert compiled.exists()
