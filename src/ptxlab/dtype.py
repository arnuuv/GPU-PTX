"""Scalar types used by the IR and the PTX subset."""

from __future__ import annotations

from enum import Enum


class DType(Enum):
    F32 = "f32"
    F16 = "f16"
    S32 = "s32"
    U32 = "u32"
    U64 = "u64"
    PRED = "pred"

    @property
    def size_bytes(self) -> int:
        if self is DType.PRED:
            raise ValueError("predicates are not stored in memory")
        if self is DType.U64:
            return 8
        if self is DType.F16:
            return 2
        return 4

    @property
    def ptx_reg_prefix(self) -> str:
        return {
            DType.F32: "%f",
            DType.F16: "%h",
            DType.S32: "%rs",
            DType.U32: "%r",
            DType.U64: "%rd",
            DType.PRED: "%p",
        }[self]

    @property
    def ptx_reg_kind(self) -> str:
        return {
            DType.F32: ".f32",
            DType.F16: ".f16",
            DType.S32: ".s32",
            DType.U32: ".u32",
            DType.U64: ".u64",
            DType.PRED: ".pred",
        }[self]
