"""AOT build for glm53_sparse_mla.

Built INSIDE the target container (the .so is libtorch- and Python-ABI-tagged),
then mounted read-only next to the existing patch/ directory. JIT
cpp_extension.load() is deliberately not used for deployment: it needs a
writable cache and ninja in a container whose files are read-only-mounted, and
it re-JITs on every cold start.
"""

import os

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

# sm_120a covers RTX Blackwell workstation parts, sm_121a the GB10s; compute_120
# PTX is a JIT fallback for anything else. Both are emitted into one .so so the
# same artifact runs on the dev box and the cluster.
ARCHS = os.environ.get("GLM53_ARCHS", "120a,121a").split(",")
gencode = [f"-gencode=arch=compute_{a},code=sm_{a}" for a in ARCHS]
gencode.append("-gencode=arch=compute_120,code=compute_120")

setup(
    name="glm53_sparse_mla",
    version="0.1.0",
    packages=["glm53_sparse_mla"],
    ext_modules=[
        CUDAExtension(
            name="glm53_sparse_mla._C",
            sources=["csrc/sparse_mla.cu"],
            extra_compile_args={
                "cxx": ["-O3", "-std=c++17"],
                "nvcc": [
                    "-O3",
                    "-std=c++17",
                    # --use_fast_math is deliberately ABSENT: it rewrites exp2f
                    # and breaks the numeric match with the TileLang reference.
                    "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
                    "-U__CUDA_NO_BFLOAT162_OPERATORS__",
                    "--expt-relaxed-constexpr",
                    "--expt-extended-lambda",
                    "--ptxas-options=-v,--warn-on-spills",
                    "-lineinfo",
                    "-DNDEBUG",
                ] + gencode,
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension},
    # vLLM calls load_general_plugins() in every process (engine core, worker,
    # arg parsing) BEFORE backend selection, so registering here needs no
    # patched vLLM file at all. register() is itself gated on
    # VLLM_GLM53_CUDA_SPARSE_MLA, so merely installing the package changes
    # nothing.
    entry_points={
        "vllm.general_plugins": [
            "glm53_sparse_mla = glm53_sparse_mla.backend:register",
            "glm53_moe_fix = glm53_sparse_mla.moe_fix:register",
        ],
    },
)
