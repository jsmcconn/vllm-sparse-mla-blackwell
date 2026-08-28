"""Small-footprint driver for compute-sanitizer: every mask pattern, tiny shapes."""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "_stubs")); sys.path.insert(0, os.path.dirname(HERE))
import torch
from glm53_sparse_mla.build import build
from tests.cases import cases, DIM

build("m1", ["csrc/sparse_mla_m1.cu"])
# topk=256 keeps sanitizer runtime sane while still covering 8 index tiles
for name, q, kv, idx in cases(topk=256, rows=1024):
    out = torch.empty_like(q)
    torch.ops.glm53_sparse_mla_m1.sparse_mla_m1(q, kv, idx, DIM ** -0.5, out)
    torch.cuda.synchronize()
    print("ran", name, flush=True)
print("done")
