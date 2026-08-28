import os, sys, time
HERE=os.path.dirname(os.path.abspath(__file__)); ROOT=os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT,"tests","_stubs")); sys.path.insert(0, ROOT)
import torch
from glm53_sparse_mla.build import build
build("gather_roof", ["bench/gather_roof.cu"])
def timeit(fn,w=5,n=20):
    for _ in range(w): fn()
    torch.cuda.synchronize(); t0=time.perf_counter()
    for _ in range(n): fn()
    torch.cuda.synchronize(); return (time.perf_counter()-t0)/n*1e3
T, TOPK, DIM = 2048, 2048, 512
sink=torch.zeros(8,device="cuda",dtype=torch.bfloat16)
gbytes = T*TOPK*DIM*2/1e9
print(f"gathered bytes per call: {gbytes:.2f} GB   (T={T}, topk={TOPK})")
for rows in (16384, 262144, 1048576):
    g=torch.Generator(device="cuda").manual_seed(0)
    kv=torch.randn(rows,DIM,generator=g,device="cuda").bfloat16()
    idx=torch.randint(0,rows,(T,TOPK),generator=g,device="cuda",dtype=torch.int32)
    print(f"  KV pool {rows:>8} rows = {rows*DIM*2/1e6:8.1f} MB")
    for th in (128,256,512):
        ms=timeit(lambda: torch.ops.gather_roof_lib.gather_roof(kv,idx,sink,th))
        print(f"    threads={th:<4} {ms:7.3f} ms   {gbytes/(ms/1e3):7.0f} GB/s", flush=True)
