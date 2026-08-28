"""Head-count gate: one kernel must serve TP=2/4/8 on a 64-head model.

8 heads is the interesting case: mma.m16n8k16 has a minimum M of 16, so the
kernel pads to 16 rows and must neither read stale Q from the padding rows nor
write them to the output.
"""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
import torch  # noqa: E402
import glm53_sparse_mla as G  # noqa: E402
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(HERE, "_stubs"))
from tests.cases import DIM, TOPK, cases  # noqa: E402
from tests.reference import sparse_mla_fp32  # noqa: E402

FP8_MAX = 448.0


def run(heads, dtype):
    ok = True
    worst = 0.0
    for name, q, kv, idx in cases(heads=heads):
        if dtype == "fp8":
            scale = (kv.float().abs().max() / FP8_MAX).clamp_min(1e-12)
            kvk = (kv.float() / scale).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
            kvr = (kvk.float() * scale).bfloat16()
            out = G.sparse_fwd(q, kvk, idx, DIM ** -0.5, kv_scale=float(scale))
        else:
            kvr = kv
            out = G.sparse_fwd(q, kv, idx, DIM ** -0.5)
        torch.cuda.synchronize()
        ref, _ = sparse_mla_fp32(q, kvr, idx, DIM ** -0.5)
        got, ref = out.float(), ref.float()
        rel = (got - ref).abs().max().item() / max(ref.abs().max().item(), 1e-6)
        cos = torch.nn.functional.cosine_similarity(
            got.reshape(-1, DIM), ref.reshape(-1, DIM), dim=-1)
        nz = ref.reshape(-1, DIM).abs().sum(-1) > 0
        cmin = cos[nz].min().item() if nz.any() else 1.0
        # see run_installed.py for why the relative bound is device-tolerant
        good = bool(torch.isfinite(got).all()) and rel <= 6e-3 and cmin >= 0.9999
        worst = max(worst, rel)
        ok &= good
        if not good:
            print(f"    FAIL {name}: rel={rel:.3e} cos={cmin:.6f}")
    return ok, worst


def main():
    p = torch.cuda.get_device_properties(0)
    print(f"device={p.name} cc={p.major}.{p.minor}")
    allok = True
    for heads, tp in ((32, 2), (16, 4), (8, 8)):
        for dt in ("bf16", "fp8"):
            ok, worst = run(heads, dt)
            allok &= ok
            print(f"  heads={heads:<3} (TP={tp})  {dt:<5} worst_rel={worst:.3e} "
                  f"{'OK' if ok else 'FAIL'}")
    print()
    print("HEAD-COUNT GATE:", "PASS" if allok else "FAIL")
    return 0 if allok else 1


if __name__ == "__main__":
    raise SystemExit(main())
