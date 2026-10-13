# NVIDIA execution

PTXLab's CPU interpreter and its analytical timing model do not launch a GPU. Real execution goes through the CUDA driver in `ptxlab.cuda_host`.

## This development machine

The repository is developed on macOS. `libcuda` is not present, and neither `ptxas` nor `nvcc` is on `PATH`. `ptxlab validate examples/vector_add.py --backend cuda` exits with status 2 and explains that the driver library was not found. That is a skip, not a failed kernel and not a simulated benchmark.

## Linux host with a GPU

Install a driver that supports the GPU and the CUDA toolkit so `ptxas` is on `PATH`. Then:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
ptxlab validate examples/vector_add.py --backend cuda
ptxlab run examples/vector_add.py --backend cuda
ptxlab bench --backend cuda
```

`validate` and `run` compile the example, launch it with `cuLaunchKernel`, copy the arrays back, and compare them with the example's NumPy reference. A mismatch is a non-zero exit. `bench --backend cuda` refuses to run when no device is present; it does not substitute interpreter timings.

`ptxas -arch=<target>` is used when `ptxas` exists. The architecture comes from the PTX `.target` directive unless `--arch sm_XX` is passed. Without `ptxas`, the driver JITs the PTX string. Both paths require `libcuda` and a device.

Compute Sanitizer is not wrapped. When it is installed, a manual check is:

```bash
compute-sanitizer python -m ptxlab run examples/vector_add.py --backend cuda
```

## Container

`docker/Dockerfile` starts from `nvidia/cuda:12.4.1-devel-ubuntu22.04`, which contains `ptxas`. It does not contain a GPU. Launching the image still needs a driver on the host:

```bash
docker build -f docker/Dockerfile -t ptxlab .
docker run --gpus all ptxlab
```

The image was not built or run as part of the macOS development of this tree.
