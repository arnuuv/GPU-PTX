"""Optional NVIDIA driver launch for PTX emitted by PTXLab.

This module never invents a toolkit. ``detect`` loads ``libcuda`` and asks the
driver how many devices it has. ``launch`` is called only when that succeeds.
On macOS, and on any host without the NVIDIA driver, ``detect().available`` is
false and ``launch`` raises ``CudaUnavailable``.

Kernel arguments use the CUDA Driver API ``kernelParams`` array: one pointer
per ``.param``, aimed at a naturally aligned value. Pointer parameters are
device pointers (``CUdeviceptr``). ``ptxas`` is used when it is on ``PATH``;
otherwise the driver JITs the PTX text.
"""

from __future__ import annotations

import ctypes
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ptxlab.errors import CudaLaunchError, CudaUnavailable
from ptxlab.parse import Program

_LIB_NAMES = ("libcuda.so.1", "libcuda.so", "libcuda.dylib")


@dataclass(frozen=True)
class CudaStatus:
    available: bool
    reason: str
    ptxas: str | None
    nvcc: str | None
    driver_library: str | None
    device_count: int
    device_name: str | None
    driver_version: int | None

    @property
    def kind(self) -> str:
        return "nvidia-driver" if self.available else "unavailable"


def detect() -> CudaStatus:
    ptxas = shutil.which("ptxas")
    nvcc = shutil.which("nvcc")
    loaded = _load_driver()
    if loaded is None:
        return CudaStatus(
            False,
            "NVIDIA driver library libcuda was not found. This host cannot launch CUDA kernels.",
            ptxas,
            nvcc,
            None,
            0,
            None,
            None,
        )
    lib, path = loaded
    try:
        _check(lib, lib.cuInit(0), "cuInit")
        count = ctypes.c_int()
        _check(lib, lib.cuDeviceGetCount(ctypes.byref(count)), "cuDeviceGetCount")
        version = ctypes.c_int()
        _check(lib, lib.cuDriverGetVersion(ctypes.byref(version)), "cuDriverGetVersion")
        if count.value < 1:
            return CudaStatus(
                False,
                "The CUDA driver is installed and reports zero devices.",
                ptxas,
                nvcc,
                path,
                0,
                None,
                version.value,
            )
        device = ctypes.c_int()
        _check(lib, lib.cuDeviceGet(ctypes.byref(device), 0), "cuDeviceGet")
        name = ctypes.create_string_buffer(256)
        _check(lib, lib.cuDeviceGetName(name, 256, device), "cuDeviceGetName")
    except CudaLaunchError as exc:
        return CudaStatus(False, str(exc), ptxas, nvcc, path, 0, None, None)
    return CudaStatus(
        True,
        "CUDA driver and at least one device are available.",
        ptxas,
        nvcc,
        path,
        count.value,
        name.value.decode(),
        version.value,
    )


def launch(
    program: Program,
    ptx: str,
    *,
    grid: tuple[int, int, int],
    block: tuple[int, int, int],
    args: dict[str, object],
) -> float:
    """Run ``program`` on device 0 and copy array arguments back into ``args``.

    ``args`` must already hold the host arrays and scalars. Array buffers are
    overwritten with the device results. This is a hardware launch, not the
    CPU interpreter.
    """

    status = detect()
    if not status.available:
        raise CudaUnavailable(status.reason)
    lib, _path = _load_driver()
    assert lib is not None
    image = _module_image(ptx, program.target, status.ptxas)
    device = ctypes.c_int()
    context = ctypes.c_void_p()
    module = ctypes.c_void_p()
    function = ctypes.c_void_p()
    _check(lib, lib.cuDeviceGet(ctypes.byref(device), 0), "cuDeviceGet")
    _check(lib, lib.cuCtxCreate(ctypes.byref(context), 0, device), "cuCtxCreate")
    allocations: list[int] = []
    try:
        image_buf = ctypes.create_string_buffer(image)
        _check(lib, lib.cuModuleLoadData(ctypes.byref(module), image_buf), "cuModuleLoadData")
        _check(
            lib,
            lib.cuModuleGetFunction(ctypes.byref(function), module, program.name.encode()),
            "cuModuleGetFunction",
        )
        storage, pointers, allocations = _bind(lib, program, args)
        shared = sum(item.nbytes for item in program.shared)
        params = (ctypes.c_void_p * len(pointers))(*pointers)
        started = time.perf_counter()
        _check(
            lib,
            lib.cuLaunchKernel(
                function,
                grid[0],
                grid[1],
                grid[2],
                block[0],
                block[1],
                block[2],
                shared,
                None,
                params,
                None,
            ),
            "cuLaunchKernel",
        )
        _check(lib, lib.cuCtxSynchronize(), "cuCtxSynchronize")
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        _copy_back(lib, program, args, storage)
    finally:
        for device_ptr in allocations:
            lib.cuMemFree(ctypes.c_uint64(device_ptr))
        if module:
            lib.cuModuleUnload(module)
        if context:
            lib.cuCtxDestroy(context)
    return elapsed_ms


def pack_scalar(kind: str, value: object) -> ctypes._SimpleCData:
    """Host storage for one kernel parameter. Tested without a GPU."""

    if kind == "u32":
        return ctypes.c_uint32(int(value) & 0xFFFFFFFF)
    if kind == "s32":
        return ctypes.c_int32(int(value))
    if kind == "f32":
        return ctypes.c_float(float(np.float32(value)))
    if kind == "u64":
        return ctypes.c_uint64(int(value))
    raise CudaLaunchError(f"cannot pack parameter kind {kind}")


def _bind(lib, program: Program, args: dict[str, object]):
    storage: list[tuple[str, ctypes._SimpleCData, np.ndarray | None, int]] = []
    pointers: list[int] = []
    allocations: list[int] = []
    for param in program.params:
        value = args[param.name]
        if param.kind == "u64":
            if not isinstance(value, np.ndarray):
                raise CudaLaunchError(f"{param.name} expects an array")
            flat = np.ascontiguousarray(value)
            ptr = ctypes.c_uint64()
            _check(lib, lib.cuMemAlloc(ctypes.byref(ptr), flat.nbytes), "cuMemAlloc")
            allocations.append(int(ptr.value))
            _check(
                lib,
                lib.cuMemcpyHtoD(ptr, flat.ctypes.data, flat.nbytes),
                "cuMemcpyHtoD",
            )
            held = ctypes.c_uint64(ptr.value)
            storage.append((param.name, held, flat, int(ptr.value)))
        else:
            held = pack_scalar(param.kind, value)
            storage.append((param.name, held, None, 0))
        pointers.append(ctypes.addressof(storage[-1][1]))
    return storage, pointers, allocations


def _copy_back(lib, program: Program, args: dict[str, object], storage) -> None:
    for name, _held, flat, device_ptr in storage:
        if flat is None:
            continue
        _check(lib, lib.cuMemcpyDtoH(flat.ctypes.data, ctypes.c_uint64(device_ptr), flat.nbytes), "cuMemcpyDtoH")
        host = args[name]
        if isinstance(host, np.ndarray):
            host[...] = flat.reshape(host.shape)


def _module_image(ptx: str, target: str, ptxas: str | None) -> bytes:
    if ptxas is None:
        return ptx.encode() + b"\0"
    with tempfile.TemporaryDirectory() as directory:
        source = Path(directory) / "kernel.ptx"
        binary = Path(directory) / "kernel.cubin"
        source.write_text(ptx, encoding="utf-8")
        result = subprocess.run(
            [ptxas, f"-arch={target}", str(source), "-o", str(binary)],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            raise CudaLaunchError(result.stderr.strip() or "ptxas failed")
        return binary.read_bytes()


def _load_driver():
    for name in _LIB_NAMES:
        try:
            lib = ctypes.CDLL(name)
        except OSError:
            continue
        _bind_signatures(lib)
        return lib, name
    return None


def _bind_signatures(lib) -> None:
    lib.cuInit.argtypes = [ctypes.c_uint]
    lib.cuInit.restype = ctypes.c_int
    lib.cuDeviceGetCount.argtypes = [ctypes.POINTER(ctypes.c_int)]
    lib.cuDeviceGetCount.restype = ctypes.c_int
    lib.cuDriverGetVersion.argtypes = [ctypes.POINTER(ctypes.c_int)]
    lib.cuDriverGetVersion.restype = ctypes.c_int
    lib.cuDeviceGet.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int]
    lib.cuDeviceGet.restype = ctypes.c_int
    lib.cuDeviceGetName.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.c_int]
    lib.cuDeviceGetName.restype = ctypes.c_int
    lib.cuCtxCreate.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_uint, ctypes.c_int]
    lib.cuCtxCreate.restype = ctypes.c_int
    lib.cuCtxDestroy.argtypes = [ctypes.c_void_p]
    lib.cuCtxDestroy.restype = ctypes.c_int
    lib.cuModuleLoadData.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p]
    lib.cuModuleLoadData.restype = ctypes.c_int
    lib.cuModuleGetFunction.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p, ctypes.c_char_p]
    lib.cuModuleGetFunction.restype = ctypes.c_int
    lib.cuModuleUnload.argtypes = [ctypes.c_void_p]
    lib.cuModuleUnload.restype = ctypes.c_int
    lib.cuMemAlloc.argtypes = [ctypes.POINTER(ctypes.c_uint64), ctypes.c_size_t]
    lib.cuMemAlloc.restype = ctypes.c_int
    lib.cuMemFree.argtypes = [ctypes.c_uint64]
    lib.cuMemFree.restype = ctypes.c_int
    lib.cuMemcpyHtoD.argtypes = [ctypes.c_uint64, ctypes.c_void_p, ctypes.c_size_t]
    lib.cuMemcpyHtoD.restype = ctypes.c_int
    lib.cuMemcpyDtoH.argtypes = [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_size_t]
    lib.cuMemcpyDtoH.restype = ctypes.c_int
    lib.cuLaunchKernel.argtypes = [
        ctypes.c_void_p,
        ctypes.c_uint,
        ctypes.c_uint,
        ctypes.c_uint,
        ctypes.c_uint,
        ctypes.c_uint,
        ctypes.c_uint,
        ctypes.c_uint,
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.c_void_p,
    ]
    lib.cuLaunchKernel.restype = ctypes.c_int
    lib.cuCtxSynchronize.argtypes = []
    lib.cuCtxSynchronize.restype = ctypes.c_int
    lib.cuGetErrorName.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_char_p)]
    lib.cuGetErrorName.restype = ctypes.c_int
    lib.cuGetErrorString.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_char_p)]
    lib.cuGetErrorString.restype = ctypes.c_int


def _check(lib, code: int, what: str) -> None:
    if code == 0:
        return
    name = ctypes.c_char_p()
    text = ctypes.c_char_p()
    if lib.cuGetErrorName(code, ctypes.byref(name)) != 0:
        raise CudaLaunchError(f"{what} failed with CUDA error {code}")
    lib.cuGetErrorString(code, ctypes.byref(text))
    detail = text.value.decode() if text.value else ""
    label = name.value.decode() if name.value else str(code)
    raise CudaLaunchError(f"{what} failed: {label} {detail}".strip())
