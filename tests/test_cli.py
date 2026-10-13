"""Command line compile/simulate and the benchmark script."""

from pathlib import Path

import numpy as np

from ptxlab.cli import main
from ptxlab.driver import compile_kernel
from ptxlab.kernels import reduction, saxpy, tiled_matmul, vector_add, vector_mul
from ptxlab.parse import parse


def test_compile_and_simulate_vector_add(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    source = tmp_path / "vector_add.py"
    source.write_text(
        "\n".join(
            [
                "import numpy as np",
                "from ptxlab.kernels import vector_add",
                "from ptxlab.reference import vector_add as ref",
                "def build_kernel():",
                "    return vector_add()",
                "def sample():",
                "    a = np.arange(32, dtype=np.float32)",
                "    b = np.ones(32, np.float32)",
                "    c = np.zeros(32, np.float32)",
                "    return (1, 1, 1), (32, 1, 1), {'a': a, 'b': b, 'c': c, 'n': 32}, {'c': ref(a, b)}",
                "",
            ]
        ),
        encoding="utf-8",
    )
    assert main(["compile", str(source)]) == 0
    ptx = tmp_path / "vector_add.ptx"
    assert ptx.exists()
    assert ".visible .entry vector_add" in ptx.read_text(encoding="utf-8")
    assert main(["simulate", str(ptx)]) == 0


def test_benchmark_script_prints_estimates(capsys) -> None:
    import importlib.util

    path = Path("benchmarks/bench.py")
    spec = importlib.util.spec_from_file_location("ptxlab_bench", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.main()
    out = capsys.readouterr().out
    assert "not GPU measurements" in out
    assert "vector_add" in out
    assert "est_cycles" in out


def test_supported_instruction_regression() -> None:
    from ptxlab.driver import execute_kernel
    from ptxlab.dtype import DType
    from ptxlab.dsl import KernelBuilder

    k = KernelBuilder("ops")
    src = k.array("src", DType.S32, ("n",))
    dst = k.array("dst", DType.S32, ("n",))
    flags = k.array("flags", DType.U32, ("n",))
    fa = k.array("fa", DType.F32, ("n",))
    fb = k.array("fb", DType.F32, ("n",))
    fc = k.array("fc", DType.F32, ("n",))
    length = k.scalar("n", DType.U32)
    i = k.let("i", k.gid("x"))
    with k.when(k.land(i < length, k.lor(i == k.u32(0), k.lnot(i < k.u32(2))))):
        k.store(dst, (i,), k.mad_lo(k.load(src, (i,)) - k.s32(1), k.s32(3), k.s32(1)))
    with k.when(k.land(i < length, i >= k.u32(1))):
        k.store(flags, (i,), k.shr(i, 1))
    with k.when(i < length):
        k.store(fc, (i,), k.load(fa, (i,)) - k.load(fb, (i,)))
    ops_kernel = k.build()
    texts = [
        compile_kernel(vector_add()),
        compile_kernel(vector_mul()),
        compile_kernel(saxpy()),
        compile_kernel(reduction(32)),
        compile_kernel(tiled_matmul(8)),
        compile_kernel(ops_kernel),
    ]
    ops: set[str] = set()
    for text in texts:
        ops.update(inst.op for inst in parse(text).insts)
    required = {
        "mov.u32",
        "mov.f32",
        "mov.s32",
        "add.u32",
        "add.u64",
        "add.rn.f32",
        "sub.rn.f32",
        "sub.s32",
        "mul.lo.u32",
        "mul.rn.f32",
        "mul.wide.u32",
        "mad.lo.s32",
        "shr.u32",
        "setp.lt.u32",
        "setp.ge.u32",
        "setp.eq.u32",
        "and.pred",
        "or.pred",
        "not.pred",
        "bra",
        "ld.param.u32",
        "ld.param.u64",
        "ld.param.f32",
        "ld.global.f32",
        "ld.global.s32",
        "st.global.f32",
        "st.global.s32",
        "ld.shared.f32",
        "st.shared.f32",
        "bar.sync",
        "ret",
    }
    assert required <= ops
    n = 4
    execute_kernel(
        ops_kernel,
        grid=(1, 1, 1),
        block=(32, 1, 1),
        args={
            "src": np.arange(n, dtype=np.int32),
            "dst": np.zeros(n, np.int32),
            "flags": np.zeros(n, np.uint32),
            "fa": np.arange(n, dtype=np.float32),
            "fb": np.ones(n, np.float32),
            "fc": np.zeros(n, np.float32),
            "n": n,
        },
    )
