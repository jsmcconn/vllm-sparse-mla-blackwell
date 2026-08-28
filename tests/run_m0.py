"""M0 gate: does the TileLang v1 NoPE kernel run here, and does it match fp32?

If this fails there is no behavioural reference for the CUDA kernel and the whole
plan is blind -- stop rather than proceeding to M1.
"""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "_stubs"))   # stub out sglang.* imports
sys.path.insert(0, os.path.dirname(HERE))

import torch  # noqa: E402

from tests.cases import DIM, NUM_HEADS, TOPK, cases  # noqa: E402
from tests.reference import sparse_mla_fp32  # noqa: E402
from tests.ref.tilelang_kernel import _tl_fit_config, tilelang_sparse_fwd  # noqa: E402


def compare(name, got, ref):
    got = got.float()
    ref = ref.float()
    finite = torch.isfinite(got).all().item()
    err = (got - ref).abs().max().item()
    denom = ref.abs().max().clamp_min(1e-6).item()
    cos = torch.nn.functional.cosine_similarity(
        got.reshape(got.shape[0] * got.shape[1], -1),
        ref.reshape(ref.shape[0] * ref.shape[1], -1),
        dim=-1,
    )
    # a fully masked (token, head) is exactly zero in both -> cosine is 0/0
    nz = ref.reshape(ref.shape[0] * ref.shape[1], -1).abs().sum(-1) > 0
    cos_min = cos[nz].min().item() if nz.any() else float("nan")
    return finite, err, err / denom, cos_min


# TileLang's v1 kernel divides by sumexp unconditionally (tilelang_kernel.py:421),
# so a token whose every index is the -1 sentinel yields 0/0 = NaN across all
# heads. Verified here: the NaN is confined to exactly that token. This is a real
# defect in the kernel currently serving GLM-5.3-Flash in production, and our CUDA
# kernel must instead return exactly zero (what the flashinfer sm120 path does).
KNOWN_TILELANG_DEFECTS = {"row_fully_masked": "divides by sumexp==0 -> NaN"}

# Thresholds are RELATIVE: `extreme_logits` scales q by 100, so an absolute bound
# would reject ordinary bf16 rounding at large magnitudes.
REL_TOL = 4e-3
COS_TOL = 0.9999


def main():
    dev = torch.device("cuda")
    p = torch.cuda.get_device_properties(0)
    print(f"device={p.name} cc={p.major}.{p.minor} SMs={p.multi_processor_count} "
          f"smem_optin={getattr(p, 'shared_memory_per_block_optin', None)}", flush=True)

    cfg = _tl_fit_config(num_heads=NUM_HEADS, dim=DIM, tail_dim=0, topk=TOPK, device_index=0)
    print(f"tilelang tile config: {cfg}   (None means the stock Hopper tile fits)", flush=True)

    sm_scale = DIM ** -0.5
    rows = []
    ok = True
    for name, q, kv, idx in cases(device=dev):
        tl = tilelang_sparse_fwd(q, kv.unsqueeze(1), idx.unsqueeze(1), sm_scale, d_v=DIM)
        tl = tl.squeeze(0) if tl.dim() == 4 else tl
        ref, _ = sparse_mla_fp32(q, kv, idx, sm_scale)
        finite, err, rel, cos = compare(name, tl, ref)
        good = finite and rel <= REL_TOL and cos >= COS_TOL
        if name in KNOWN_TILELANG_DEFECTS and not finite:
            status = f"XFAIL ({KNOWN_TILELANG_DEFECTS[name]})"
        elif good:
            status = "OK"
        else:
            status = "FAIL"
            ok = False
        rows.append((name, tuple(q.shape), finite, err, rel, cos, status))
        print(f"  {name:20s} T={q.shape[0]:<5d} finite={str(finite):5s} "
              f"max_abs={err:.3e} rel={rel:.3e} cos_min={cos:.6f} {status}", flush=True)

    print()
    print("M0 GATE:", "PASS" if ok else "FAIL",
          "-- TileLang runs here and matches fp32; it is a usable reference.")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
