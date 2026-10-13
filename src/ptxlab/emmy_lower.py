"""Lower a narrow slice of Emmy's real tensor IR into a PTXLab kernel.

Requires the Emmy source tree on ``PYTHONPATH``. The supported commit is
``02cab261fdd89fd261441f2d02ddb400e94ead3a``. This does not import Emmy at
PTXLab import time, and it does not modify Emmy.

Supported graphs have one compute node:

* ``ElementwiseOp`` named ``add``, ``multiply``, or ``subtract``, on two
  rank-1 ``f32`` tensors with the same static length.
* ``ReduceOp`` named ``sum`` on axis 0 of one rank-1 ``f32`` tensor whose
  length is a power of two in 1..1024. The device writes one value.

Symbolic shapes, other dtypes, and every other op raise ``EmmyLowerError``.
"""

from __future__ import annotations

from ptxlab.dsl import Kernel
from ptxlab.dtype import DType
from ptxlab.errors import EmmyLowerError
from ptxlab.kernels import reduction

_ELEMENTWISE = {"add": "add", "multiply": "mul", "subtract": "sub"}


def lower_emmy_graph(graph) -> tuple[Kernel, dict[str, object]]:
    compute = []
    for node_id in graph.topological_order():
        node = graph.nodes[node_id]
        if type(node.op).__name__ in {"InputOp", "ConstantOp"}:
            continue
        compute.append(node)
    if len(compute) != 1:
        raise EmmyLowerError(f"expected one compute node, found {len(compute)}")
    node = compute[0]
    kind = type(node.op).__name__
    if kind == "ElementwiseOp":
        return _elementwise(graph, node), {"kind": "elementwise"}
    if kind == "ReduceOp":
        kernel, launch = _reduce(graph, node)
        return kernel, launch
    raise EmmyLowerError(f"{kind} is not lowered")


def _elementwise(graph, node) -> Kernel:
    from ptxlab.dsl import KernelBuilder

    name = node.op.name
    if name not in _ELEMENTWISE:
        raise EmmyLowerError(f"elementwise op {name!r} is not lowered")
    if len(node.inputs) != 2:
        raise EmmyLowerError("binary elementwise ops need two inputs")
    left = _vector(graph, node.inputs[0])
    right = _vector(graph, node.inputs[1])
    out = _vector_tensor(node.output)
    if left != right or left != out:
        raise EmmyLowerError("elementwise inputs and output must share one static f32 length")
    builder = KernelBuilder(f"emmy_{name}")
    a = builder.array("a", DType.F32, ("n",))
    b = builder.array("b", DType.F32, ("n",))
    c = builder.array("c", DType.F32, ("n",))
    n = builder.scalar("n", DType.U32)
    index = builder.let("i", builder.gid("x"))
    loaded_a = builder.load(a, (index,))
    loaded_b = builder.load(b, (index,))
    if name == "add":
        value = loaded_a + loaded_b
    elif name == "multiply":
        value = loaded_a * loaded_b
    else:
        value = loaded_a - loaded_b
    with builder.when(index < n):
        builder.store(c, (index,), value)
    kernel = builder.build()
    kernel.expect_block = None
    return kernel


def _reduce(graph, node) -> tuple[Kernel, dict[str, object]]:
    if node.op.name != "sum" or int(node.op.axis) != 0:
        raise EmmyLowerError("only a sum on axis 0 is lowered")
    if len(node.inputs) != 1:
        raise EmmyLowerError("reduction expects one input")
    length = _vector(graph, node.inputs[0])
    if length > 1024 or length & (length - 1):
        raise EmmyLowerError("sum length must be a power of two up to 1024 so one block can reduce it")
    kernel = reduction(length)
    return kernel, {"kind": "reduce", "block": length, "grid": (1, 1, 1)}


def _vector(graph, name: str) -> int:
    node = graph.nodes[name]
    return _vector_tensor(node.output)


def _vector_tensor(tensor) -> int:
    if getattr(tensor.dtype, "name", None) != "f32":
        raise EmmyLowerError(f"only f32 is lowered, got {getattr(tensor.dtype, 'name', tensor.dtype)}")
    if len(tensor.shape) != 1:
        raise EmmyLowerError("only rank-1 tensors are lowered")
    dim = tensor.shape[0]
    if not getattr(dim, "is_static", False):
        raise EmmyLowerError("symbolic dimensions are not lowered")
    return int(dim.as_static())
