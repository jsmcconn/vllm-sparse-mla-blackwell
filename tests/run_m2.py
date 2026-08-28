"""M2 gate: tensor-core kernel must match fp32 exactly as M1 did."""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "_stubs"))
sys.path.insert(0, os.path.dirname(HERE))

import torch  # noqa: E402

from glm53_sparse_mla.build import build  # noqa: E402
from tests.cases import DIM, cases  # noqa: E402
from tests.reference import sparse_mla_fp32  # noqa: E402

REL_TOL = 4e-3
COS_TOL = 0.9999


def main():
    build("m2", ["csrc/sparse_mla_m2.cu"])
    p = torch.cuda.get_device_properties(0)
    print(f"device={p.name} cc={p.major}.{p.minor} SMs={p.multi_processor_count}", flush=True)

    sm_scale = DIM ** -0.5
    ok = True
    for name, q, kv, idx in cases():
        out = torch.empty_like(q)
        torch.ops.glm53_sparse_mla_m2.sparse_mla_m2(q, kv, idx, sm_scale, out)
        torch.cuda.synchronize()
        ref, _ = sparse_mla_fp32(q, kv, idx, sm_scale)

        got = out.float()
        ref = ref.float()
        finite = bool(torch.isfinite(got).all())
        err = (got - ref).abs().max().item()
        rel = err / max(ref.abs().max().item(), 1e-6)
        flat_g = got.reshape(-1, DIM)
        flat_r = ref.reshape(-1, DIM)
        nz = flat_r.abs().sum(-1) > 0
        cos = torch.nn.functional.cosine_similarity(flat_g, flat_r, dim=-1)
        cos_min = cos[nz].min().item() if nz.any() else 1.0

        good = finite and rel <= REL_TOL and cos_min >= COS_TOL
        # the fully-masked case additionally asserts the return-zero decision
        if name == "row_fully_masked":
            good = good and bool((out[2] == 0).all())
        ok &= good
        print(f"  {name:20s} T={q.shape[0]:<5d} finite={str(finite):5s} "
              f"max_abs={err:.3e} rel={rel:.3e} cos_min={cos_min:.6f} "
              f"{'OK' if good else 'FAIL'}", flush=True)

    print()
    print("M2 GATE:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
