import os, sys, time
HERE=os.path.dirname(os.path.abspath(__file__)); ROOT=os.path.dirname(HERE)
import torch
import glm53_sparse_mla as G
sys.path.insert(0, ROOT)
from tests.cases import DIM, NUM_HEADS, TOPK
FP8_MAX=448.0
def timeit(fn,w=8,n=25):
    for _ in range(w): fn()
    torch.cuda.synchronize(); t0=time.perf_counter()
    for _ in range(n): fn()
    torch.cuda.synchronize(); return (time.perf_counter()-t0)/n*1e3
sm=DIM**-0.5
print(f"{'T':>6} {'rows':>9} {'bf16_ms':>9} {'fp8_ms':>9} {'speedup':>8}")
for rows in (16384, 262144):
    for T in (128, 2048):
        g=torch.Generator(device='cuda').manual_seed(T)
        q=torch.randn(T,NUM_HEADS,DIM,generator=g,device='cuda').bfloat16()
        kv=torch.randn(rows,DIM,generator=g,device='cuda').bfloat16()
        idx=torch.randint(0,rows,(T,TOPK),generator=g,device='cuda',dtype=torch.int32)
        out=torch.empty_like(q)
        scale=(kv.float().abs().max()/FP8_MAX).clamp_min(1e-12)
        kv8=(kv.float()/scale).clamp(-FP8_MAX,FP8_MAX).to(torch.float8_e4m3fn)
        b=timeit(lambda: G.sparse_fwd(q,kv,idx,sm,1.0,out))
        f=timeit(lambda: G.sparse_fwd(q,kv8,idx,sm,float(scale),out))
        print(f"{T:>6} {rows:>9} {b:>9.3f} {f:>9.3f} {b/f:>7.2f}x", flush=True)
