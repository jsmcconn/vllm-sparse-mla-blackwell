"""Wall-clock comparison: our kernel vs the TileLang v1 reference, same shapes."""
import os, sys, time
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "tests", "_stubs")); sys.path.insert(0, ROOT)
import torch
from glm53_sparse_mla.build import build
from tests.cases import DIM, NUM_HEADS, TOPK
from tests.ref.tilelang_kernel import tilelang_sparse_fwd

def timeit(fn, warmup=5, iters=20):
    for _ in range(warmup): fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters): fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1e3   # ms

def main():
    build("m1", ["csrc/sparse_mla_m1.cu"])
    build("m2", ["csrc/sparse_mla_m2.cu"])
    build("m3", ["csrc/sparse_mla_m3.cu"])
    build("m4", ["csrc/sparse_mla_m4.cu"])
    build("m5", ["csrc/sparse_mla_m5.cu"])
    dev = "cuda"; sm = DIM ** -0.5
    print(f"{'T':>6} {'tilelang_ms':>12} {'m1_ms':>10} {'m2_ms':>10} {'m3_ms':>10} {'m4_ms':>10} {'m5_ms':>10} {'tl/m5 (>1 = we win)':>22}")
    for T in (1, 128, 512, 2048, 8192):
        g = torch.Generator(device=dev).manual_seed(T)
        q = torch.randn(T, NUM_HEADS, DIM, generator=g, device=dev).bfloat16()
        kv = torch.randn(16384, DIM, generator=g, device=dev).bfloat16()
        idx = torch.randint(0, 16384, (T, TOPK), generator=g, device=dev, dtype=torch.int32)
        out = torch.empty_like(q)
        kv3, idx3 = kv.unsqueeze(1), idx.unsqueeze(1)
        tl = timeit(lambda: tilelang_sparse_fwd(q, kv3, idx3, sm, d_v=DIM))
        m1 = timeit(lambda: torch.ops.glm53_sparse_mla_m1.sparse_mla_m1(q, kv, idx, sm, out))
        m2 = timeit(lambda: torch.ops.glm53_sparse_mla_m2.sparse_mla_m2(q, kv, idx, sm, out))
        m3 = timeit(lambda: torch.ops.glm53_sparse_mla_m3.sparse_mla_m3(q, kv, idx, sm, out))
        m4 = timeit(lambda: torch.ops.glm53_sparse_mla_m4.sparse_mla_m4(q, kv, idx, sm, out))
        m5 = timeit(lambda: torch.ops.glm53_sparse_mla_m5.sparse_mla_m5(q, kv, idx, sm, out))
        print(f"{T:>6} {tl:>12.3f} {m1:>10.3f} {m2:>10.3f} {m3:>10.3f} {m4:>10.3f} {m5:>10.3f} {tl/m5:>21.2f}x", flush=True)

if __name__ == "__main__":
    main()
