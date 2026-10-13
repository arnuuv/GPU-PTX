# Linux image with the CUDA toolkit. Running a kernel still needs
# `docker run --gpus all` on a machine whose NVIDIA driver matches this image.
# Building this file does not execute a kernel.
FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

RUN apt-get update && apt-get install -y --no-install-recommends python3 python3-pip python3-venv \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/ptxlab
COPY . .
RUN python3 -m pip install --no-cache-dir -e ".[dev]"

CMD ["ptxlab", "validate", "examples/vector_add.py", "--backend", "cuda"]
