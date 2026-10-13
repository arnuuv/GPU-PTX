"""Lower real Emmy tensor ops. Skips when the Emmy tree is not importable."""

import numpy as np
import pytest

emmy = pytest.importorskip("emmy")
from emmy.compiler.graph import Graph
from emmy.compiler.ir.base import InputOp
from emmy.compiler.ir.tensor.ir import ElementwiseOp, ReduceOp
from emmy.compiler.tensor import Tensor

from ptxlab.driver import execute_kernel
from ptxlab.emmy_lower import lower_emmy_graph
from ptxlab.errors import EmmyLowerError
from ptxlab.reference import block_reduction


def _graph_add(length: int) -> Graph:
    graph = Graph()
    graph.add_node(InputOp(), [], output=Tensor("a", (length,), "f32"))
    graph.add_node(InputOp(), [], output=Tensor("b", (length,), "f32"))
    graph.add_node(ElementwiseOp("add"), ["a", "b"], output=Tensor("c", (length,), "f32"))
    graph.inputs = ["a", "b"]
    graph.outputs = ["c"]
    return graph


def test_emmy_elementwise_add_matches_numpy() -> None:
    kernel, info = lower_emmy_graph(_graph_add(32))
    assert info["kind"] == "elementwise"
    a = np.arange(32, dtype=np.float32)
    b = np.ones(32, np.float32)
    c = np.zeros(32, np.float32)
    execute_kernel(kernel, grid=(1, 1, 1), block=(32, 1, 1), args={"a": a, "b": b, "c": c, "n": 32})
    np.testing.assert_array_equal(c, a + b)


def test_emmy_sum_matches_the_block_tree() -> None:
    graph = Graph()
    graph.add_node(InputOp(), [], output=Tensor("src", (32,), "f32"))
    graph.add_node(ReduceOp("sum", axis=0), ["src"], output=Tensor("out", (1,), "f32"))
    kernel, info = lower_emmy_graph(graph)
    assert info["block"] == 32
    src = np.arange(32, dtype=np.float32)
    dst = np.zeros(1, np.float32)
    execute_kernel(
        kernel,
        grid=(1, 1, 1),
        block=(32, 1, 1),
        args={"src": src, "dst": dst, "n": 32, "blocks": 1},
    )
    np.testing.assert_array_equal(dst, block_reduction(src, 32))


def test_emmy_tanh_is_rejected() -> None:
    graph = Graph()
    graph.add_node(InputOp(), [], output=Tensor("a", (8,), "f32"))
    graph.add_node(ElementwiseOp("tanh"), ["a"], output=Tensor("b", (8,), "f32"))
    with pytest.raises(EmmyLowerError):
        lower_emmy_graph(graph)
