"""Single-pass block reduction. Each block writes one partial sum."""

import numpy as np

from ptxlab.kernels import reduction
from ptxlab.reference import block_reduction

BLOCK = 32


def build_kernel():
    return reduction(BLOCK)


def sample():
    rng = np.random.default_rng(3)
    n = 80
    src = rng.standard_normal(n).astype(np.float32)
    blocks = (n + BLOCK - 1) // BLOCK
    dst = np.zeros(blocks, np.float32)
    args = {"src": src, "dst": dst, "n": n, "blocks": blocks}
    return (blocks, 1, 1), (BLOCK, 1, 1), args, {"dst": block_reduction(src, BLOCK)}
