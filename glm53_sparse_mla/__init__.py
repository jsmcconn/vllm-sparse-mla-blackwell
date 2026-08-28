"""Sparse MLA (NoPE) attention for sm_120 / sm_121.

Exposes ``torch.ops.glm53_sparse_mla.sparse_fwd``. The CUDA op is registered by
importing the compiled extension; in a deployed container that is the AOT-built
``_C`` module, while local development falls back to a JIT build so the kernel
can be iterated without reinstalling.
"""

import glob
import os

import torch

__all__ = ["sparse_fwd", "HEADS_PER_RANK", "SUPPORTED_HEADS_PER_RANK", "KV_LORA_RANK"]

HEADS_PER_RANK = 32   # 64 q-heads / TP2
# 8 / 16 / 32 heads per rank, i.e. TP=8 / TP=4 / TP=2 on a 64-head model.
SUPPORTED_HEADS_PER_RANK = (8, 16, 32)
KV_LORA_RANK = 512    # head_size, since qk_rope_head_dim == 0 (NoPE)


def _load() -> None:
    """Register the CUDA op.

    The extension is TORCH_LIBRARY-only -- it has no PyInit, so a plain
    ``import`` of it fails with "dynamic module does not define module export
    function". It must be loaded with ``torch.ops.load_library``.
    """
    if hasattr(torch.ops, "glm53_sparse_mla") and hasattr(
        torch.ops.glm53_sparse_mla, "sparse_fwd"
    ):
        return

    here = os.path.dirname(os.path.abspath(__file__))
    sos = sorted(glob.glob(os.path.join(here, "_C*.so")))
    if sos:
        torch.ops.load_library(sos[0])
        return

    # Local dev fallback. Deliberately NOT wrapped in a try/except: silently
    # JIT-compiling inside a serving container -- where csrc/ is absent and the
    # filesystem is read-only -- would turn a packaging mistake into a confusing
    # runtime failure much later.
    if not os.path.exists(os.path.join(os.path.dirname(here), "csrc")):
        raise ImportError(
            "glm53_sparse_mla: no compiled _C*.so next to the package and no "
            "csrc/ to build from. The AOT build did not ship its extension."
        )
    from .build import build

    build("glm53_sparse_mla_jit", ["csrc/sparse_mla.cu"])


_load()


def sparse_fwd(q, kv, indices, sm_scale, kv_scale=1.0, out=None):
    """Gathered sparse MLA over the top-k indices.

    q       [T, 32, 512]  bfloat16
    kv      [R, 512]      bfloat16 or float8_e4m3fn, a flat row view of the paged
                          cache; serves as BOTH K and V (the absorbed-MLA identity).
                          fp8 halves the gather traffic and doubles KV capacity;
                          kv_scale is vLLM's per-tensor cache scale (1.0 for bf16).
    indices [T, topk]     int32 physical row ids, -1 is the mask sentinel
    returns [T, 32, 512]  bfloat16

    Causality is NOT applied here -- the indexer owns it, which is what lets
    physical slot ids be passed straight through. A token whose indices are all
    -1 returns exactly zero (not NaN).
    """
    if out is None:
        out = torch.empty_like(q)
    torch.ops.glm53_sparse_mla.sparse_fwd(q, kv, indices, float(sm_scale),
                                          float(kv_scale), out)
    return out


# Shape-only meta kernel so torch.compile can trace through the op instead of
# breaking the graph on it.
@torch.library.register_fake("glm53_sparse_mla::sparse_fwd")
def _(q, kv, indices, sm_scale, kv_scale, out):
    torch._check(q.dim() == 3 and kv.dim() == 2 and indices.dim() == 2)
    torch._check(out.shape == q.shape)
    return None
