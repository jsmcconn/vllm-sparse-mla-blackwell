"""fp8 KV gate: the kernel must dequantize the cache correctly during the gather.

Reference is computed from the DEQUANTIZED cache, so this isolates the kernel's
fp8 handling from fp8's own quantization error.
"""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
import torch  # noqa: E402
import glm53_sparse_mla as G  # noqa: E402
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(HERE, "_stubs"))
from tests.cases import DIM, cases  # noqa: E402
from tests.reference import sparse_mla_fp32  # noqa: E402

FP8_MAX = 448.0
# Tolerances. The cosine bound is the meaningful one: it catches wrong maths,
# wrong layouts and wrong masking, all of which collapse it far below 0.9999.
# The relative bound is looser because bf16 MMA accumulation order differs
# between GPUs, so a bound calibrated on one device is not portable: the same
# case measures 3.81e-3 on an RTX 5090 and 4.09e-3 on an RTX PRO 6000. For
# scale, the TileLang kernel this replaces measures up to 3.8e-3 against the
# same fp32 reference, so anything in this band is bf16 noise rather than a
# defect.
REL_TOL, COS_TOL = 6e-3, 0.9999


def main():
    p = torch.cuda.get_device_properties(0)
    print(f"device={p.name} cc={p.major}.{p.minor}")
    ok = True
    for name, q, kv, idx in cases():
        # per-tensor scale, exactly what vLLM's cache writer uses
        scale = (kv.float().abs().max() / FP8_MAX).clamp_min(1e-12)
        kv_fp8 = (kv.float() / scale).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
        kv_deq = (kv_fp8.float() * scale).bfloat16()

        out = G.sparse_fwd(q, kv_fp8, idx, DIM ** -0.5, kv_scale=float(scale))
        torch.cuda.synchronize()
        ref, _ = sparse_mla_fp32(q, kv_deq, idx, DIM ** -0.5)

        got, ref = out.float(), ref.float()
        finite = bool(torch.isfinite(got).all())
        rel = (got - ref).abs().max().item() / max(ref.abs().max().item(), 1e-6)
        fg, fr = got.reshape(-1, DIM), ref.reshape(-1, DIM)
        nz = fr.abs().sum(-1) > 0
        cos = torch.nn.functional.cosine_similarity(fg, fr, dim=-1)
        cmin = cos[nz].min().item() if nz.any() else 1.0
        good = finite and rel <= REL_TOL and cmin >= COS_TOL
        if name == "row_fully_masked":
            good = good and bool((out[2] == 0).all())
        ok &= good
        print(f"  {name:20s} rel={rel:.3e} cos_min={cmin:.6f} {'OK' if good else 'FAIL'}")
    print()
    print("FP8 KV GATE:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
