"""Bit-exact JSON launch records for ``ptxlab simulate``."""

from __future__ import annotations

import base64
import json
from pathlib import Path

import numpy as np

from ptxlab.errors import LaunchError

_DTYPES = {
    "f32": np.float32,
    "s32": np.int32,
    "u32": np.uint32,
}


def dump_launch(
    path: Path,
    *,
    grid: tuple[int, ...],
    block: tuple[int, ...],
    args: dict[str, object],
    compare: dict[str, np.ndarray] | None = None,
) -> None:
    payload = {
        "grid": list(grid),
        "block": list(block),
        "args": {name: _encode(value) for name, value in args.items()},
        "compare": {name: _encode(value) for name, value in (compare or {}).items()},
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def load_launch(path: Path) -> tuple[tuple[int, ...], tuple[int, ...], dict[str, object], dict[str, np.ndarray]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    args = {name: _decode(value) for name, value in payload["args"].items()}
    compare = {name: _decode(value) for name, value in payload.get("compare", {}).items()}
    return tuple(payload["grid"]), tuple(payload["block"]), args, compare


def _encode(value: object) -> dict[str, object]:
    if isinstance(value, np.ndarray):
        flat = np.ascontiguousarray(value)
        dtype = _name(flat.dtype)
        return {
            "dtype": dtype,
            "shape": list(flat.shape),
            "b64": base64.b64encode(flat.tobytes()).decode("ascii"),
        }
    if isinstance(value, (float, np.floating)):
        bits = int(np.asarray(np.float32(value)).view(np.uint32))
        return {"dtype": "f32", "bits": bits}
    if isinstance(value, (int, np.integer)) and not isinstance(value, bool):
        kind = "s32" if int(value) < 0 else "u32"
        return {"dtype": kind, "value": int(value)}
    raise LaunchError(f"cannot encode launch value {value!r}")


def _decode(value: dict[str, object]) -> object:
    dtype = str(value["dtype"])
    if "b64" in value:
        raw = base64.b64decode(str(value["b64"]))
        array = np.frombuffer(raw, dtype=_DTYPES[dtype]).copy()
        return array.reshape(tuple(int(dim) for dim in value["shape"]))  # type: ignore[arg-type]
    if dtype == "f32":
        return float(np.array(np.uint32(value["bits"])).view(np.float32))
    return int(value["value"])  # type: ignore[arg-type]


def _name(dtype: np.dtype) -> str:
    if dtype == np.float32:
        return "f32"
    if dtype == np.int32:
        return "s32"
    if dtype == np.uint32:
        return "u32"
    raise LaunchError(f"unsupported dtype {dtype}")
