"""fp32 gathered sparse-MLA attention: the ground truth for every other path.

Contract (identical to SGLang's ``sparse_attention_fwd_kernel_v1``):
  q       [T, H, D]  bf16
  kv      [R, D]     bf16   -- one MQA head; serves as BOTH K and V (absorbed MLA)
  indices [T, TOPK]  int32  -- physical row ids into ``kv``; -1 is the mask sentinel
  out     [T, H, D]  bf16

Causality is NOT applied here, deliberately: the reference kernel computes
``max_kv_i = q_i`` and never reads it (tilelang_kernel.py:361). The only mask is
``indices >= 0``. Causality is the indexer's responsibility, which is precisely
what makes the index axis layout-agnostic and lets vLLM paged slot ids be passed
straight through.
"""

import torch

LOG2E = 1.44269504


def sparse_mla_fp32(q, kv, indices, sm_scale, chunk=8):
    """Chunked over tokens; the [T, topk, D] gather is 512 MB at T=128."""
    if chunk and q.shape[0] > chunk:
        outs, lses = [], []
        for i in range(0, q.shape[0], chunk):
            o, l = _sparse_mla_fp32(q[i:i + chunk], kv, indices[i:i + chunk], sm_scale)
            outs.append(o)
            lses.append(l)
        return torch.cat(outs), torch.cat(lses)
    return _sparse_mla_fp32(q, kv, indices, sm_scale)


def _sparse_mla_fp32(q, kv, indices, sm_scale):
    """Ground truth in float32. Returns (out[T,H,D] float32, lse[T,H] float32).

    ``lse`` is returned in log2 space -- ``log2(sumexp2) + m * sm_scale * log2e`` --
    to match the split-K combine pass, which never touches natural logs.
    """
    T, H, D = q.shape
    topk = indices.shape[1]
    assert kv.shape[1] == D
    assert indices.shape[0] == T

    qf = q.float()
    kvf = kv.float()

    valid = indices >= 0                       # [T, topk]
    rows = indices.clamp_min(0).long()         # clamp: -1 would index the last row
    gathered = kvf[rows]                       # [T, topk, D]

    # scores, then mask -- masking after avoids -inf * 0 = nan
    s = torch.einsum("thd,tkd->thk", qf, gathered) * sm_scale     # [T, H, topk]
    s = s.masked_fill(~valid.unsqueeze(1), float("-inf"))

    m = s.amax(dim=-1)                                            # [T, H]
    # a fully masked row has m == -inf; make the shift finite so exp gives 0
    m_safe = torch.where(torch.isfinite(m), m, torch.zeros_like(m))
    p = torch.exp(s - m_safe.unsqueeze(-1))
    p = torch.where(valid.unsqueeze(1), p, torch.zeros_like(p))

    sumexp = p.sum(dim=-1)                                        # [T, H]
    out = torch.einsum("thk,tkd->thd", p, gathered)

    # An all-masked row yields sumexp == 0. Return exactly zero rather than NaN:
    # this matches the flashinfer sm120 kernel's documented behaviour, and NaN
    # here would poison the whole sequence.
    safe = sumexp > 0
    out = torch.where(safe.unsqueeze(-1), out / sumexp.clamp_min(1e-30).unsqueeze(-1),
                      torch.zeros_like(out))

    lse = torch.where(safe,
                      torch.log2(sumexp.clamp_min(1e-30)) + m_safe * (sm_scale * LOG2E),
                      torch.full_like(sumexp, float("-inf")))
    return out, lse
