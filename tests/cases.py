"""Case generator for the sparse-MLA kernel differential.

Each case returns (name, q, kv, indices) on the given device. The mask patterns
are the ones that actually break kernels; see the plan's case matrix.
"""

import torch

NUM_HEADS = 32          # 64 q-heads / TP2
DIM = 512               # kv_lora_rank; tail_dim == 0 (NoPE)
TOPK = 2048
NUM_ROWS = 8192         # rows in the flat KV view


def _rand(T, H, D, R, device, seed):
    g = torch.Generator(device=device).manual_seed(seed)
    q = torch.randn(T, H, D, generator=g, device=device, dtype=torch.float32).bfloat16()
    kv = torch.randn(R, D, generator=g, device=device, dtype=torch.float32).bfloat16()
    return q, kv, g


def _idx(T, topk, R, g, device):
    return torch.randint(0, R, (T, topk), generator=g, device=device, dtype=torch.int32)


def cases(device="cuda", topk=TOPK, dim=DIM, heads=NUM_HEADS, rows=NUM_ROWS):
    def base(T, seed):
        q, kv, g = _rand(T, heads, dim, rows, device, seed)
        return q, kv, g

    # 1. all indices valid
    q, kv, g = base(8, 1)
    yield "all_valid", q, kv, _idx(8, topk, rows, g, device)

    # 2/3. scattered -1 at two densities: mask handling and -inf propagation
    for frac, name in ((0.1, "mask_10pct"), (0.5, "mask_50pct")):
        q, kv, g = base(8, 2)
        idx = _idx(8, topk, rows, g, device)
        drop = torch.rand(idx.shape, generator=g, device=device) < frac
        idx = idx.masked_fill(drop, -1)
        yield name, q, kv, idx

    # 4. one whole tile masked -- reduce_max(clear=False) must leave m_i UNCHANGED
    q, kv, g = base(4, 3)
    idx = _idx(4, topk, rows, g, device)
    idx[:, 64:128] = -1
    yield "one_tile_masked", q, kv, idx

    # 5. an entire row masked -- the sumexp == 0 -> return 0, not NaN, decision
    q, kv, g = base(4, 4)
    idx = _idx(4, topk, rows, g, device)
    idx[2, :] = -1
    yield "row_fully_masked", q, kv, idx

    # 6. duplicate indices -- the kpool indexer emits them; no dedup/atomic assumption
    q, kv, g = base(4, 5)
    idx = _idx(4, topk // 8, rows, g, device).repeat(1, 8)
    yield "duplicate_indices", q, kv, idx

    # 7. degenerate: index 0 repeated, exercising the clamp path
    q, kv, g = base(2, 6)
    yield "index0_repeated", q, kv, torch.zeros(2, topk, device=device, dtype=torch.int32)

    # 8. extreme logits -- exp2 overflow and the m_i = -(2**30) init
    q, kv, g = base(4, 7)
    yield "extreme_logits", (q.float() * 100).bfloat16(), kv, _idx(4, topk, rows, g, device)

    # 9. token-count tails (grid edges, non-multiples of a warp)
    for T in (1, 7, 128):
        q, kv, g = base(T, 100 + T)
        yield f"tokens_{T}", q, kv, _idx(T, topk, rows, g, device)
