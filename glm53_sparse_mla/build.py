"""JIT build helper for local iteration (M1-M4). M5 replaces this with an AOT
wheel built inside the vLLM container -- see the plan."""

import os

import torch
from torch.utils.cpp_extension import load

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# sm_120a covers the RTX 5090 dev box, sm_121a the GB10s; compute_120 PTX is a
# JIT fallback for anything else. --use_fast_math is deliberately absent: it
# rewrites exp2f and would break the numeric match with the TileLang reference.
NVCC_FLAGS = [
    "-O3",
    "-std=c++17",
    "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
    "-U__CUDA_NO_BFLOAT162_OPERATORS__",
    "--expt-relaxed-constexpr",
    "--expt-extended-lambda",
    "--ptxas-options=-v,--warn-on-spills",
    "-lineinfo",
    "-DNDEBUG",
]


def arch_flags():
    major, minor = torch.cuda.get_device_capability(0)
    return [f"-gencode=arch=compute_{major}{minor}a,code=sm_{major}{minor}a"]


def build(name, sources, extra_nvcc=()):
    return load(
        name=name,
        sources=[os.path.join(_ROOT, s) for s in sources],
        extra_cuda_cflags=NVCC_FLAGS + arch_flags() + list(extra_nvcc),
        extra_cflags=["-O3", "-std=c++17"],
        build_directory=_ensure_build_dir(name),
        is_python_module=False,   # ops are registered via TORCH_LIBRARY, not pybind
        verbose=False,
    )


def _ensure_build_dir(name):
    d = os.path.join(_ROOT, "build", name)
    os.makedirs(d, exist_ok=True)
    return d
