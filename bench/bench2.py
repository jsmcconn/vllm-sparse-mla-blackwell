"""Focused comparison: TileLang vs the current best kernels."""
import os, sys, time
HERE=os.path.dirname(os.path.abspath(__file__)); ROOT=os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT,"tests","_stubs")); sys.path.insert(0, ROOT)
import torch
from glm53_sparse_mla.build import build
from tests.cases import DIM, NUM_HEADS, TOPK
from tests.ref.tilelang_kernel import tilelang_sparse_fwd
VARIANTS = [v for v in sys.argv[1:]] or ["m5", "m6"]
for v in VARIANTS: build(v, [f"csrc/sparse_mla_{v}.cu"])
def timeit(fn,w=8,n=30):
    for _ in range(w): fn()
    torch.cuda.synchronize(); t0=time.perf_counter()
    for _ in range(n): fn()
    torch.cuda.synchronize(); return (time.perf_counter()-t0)/n*1e3
sm=DIM**-0.5
hdr = f"{'T':>6} {'tilelang':>10}" + "".join(f"{v:>10}" for v in VARIANTS) + "".join(f"{'tl/'+v:>9}" for v in VARIANTS)
print(hdr)
for T in (1,128,512,2048,8192):
    g=torch.Generator(device="cuda").manual_seed(T)
    q=torch.randn(T,NUM_HEADS,DIM,generator=g,device="cuda").bfloat16()
    kv=torch.randn(16384,DIM,generator=g,device="cuda").bfloat16()
    idx=torch.randint(0,16384,(T,TOPK),generator=g,device="cuda",dtype=torch.int32)
    out=torch.empty_like(q); kv3,idx3=kv.unsqueeze(1),idx.unsqueeze(1)
    tl=timeit(lambda: tilelang_sparse_fwd(q,kv3,idx3,sm,d_v=DIM))
    ts=[]
    for v in VARIANTS:
        op=getattr(getattr(torch.ops, f"glm53_sparse_mla_{v}"), f"sparse_mla_{v}")
        ts.append(timeit(lambda op=op: op(q,kv,idx,sm,out)))
    print(f"{T:>6} {tl:>10.3f}" + "".join(f"{t:>10.3f}" for t in ts)
          + "".join(f"{tl/t:>8.2f}x" for t in ts), flush=True)
