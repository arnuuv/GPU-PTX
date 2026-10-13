"""Static memory classifications."""

from ptxlab.analysis import arithmetic_intensity, memory_notes
from ptxlab.dtype import DType
from ptxlab.dsl import KernelBuilder
from ptxlab.kernels import vector_add
from ptxlab.lower import lower


def test_vector_add_is_unit_stride() -> None:
    notes = [note for note in memory_notes(lower(vector_add())) if note.space == "global"]
    assert notes
    assert all(note.classification == "unit-stride" for note in notes)


def test_scaled_index_is_strided() -> None:
    k = KernelBuilder("stride")
    src = k.array("src", DType.F32, ("n",))
    dst = k.array("dst", DType.F32, ("n",))
    n = k.scalar("n", DType.U32)
    i = k.let("i", k.tid("x"))
    with k.when(i < n):
        k.store(dst, (i * k.u32(32),), k.load(src, (i * k.u32(32),)))
    notes = memory_notes(lower(k.build()))
    assert any(note.classification == "strided" for note in notes)


def test_intensity_counts_arithmetic_and_memory() -> None:
    stats = arithmetic_intensity(lower(vector_add()))
    assert stats["memory_ops"] >= 2
    assert stats["arithmetic_ops"] >= 1
