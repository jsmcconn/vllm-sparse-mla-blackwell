"""Which w13 ordering does flashinfer's cutlass NVFP4 fused MoE actually want?

vLLM applies `reorder_w1w3_to_w3w1` before calling `cutlass_fused_moe`; SGLang's
modelopt path does not, and SGLang is the one that produces correct output on
GB10 for this checkpoint. Both loaders build w13 as [gate, up]. At most one can
be right.

This isolates the question the same way the attention kernel was isolated: build
a tiny NVFP4 MoE, dequantize it to get an exact reference, then run the kernel
with [gate, up] and with [up, gate] and see which matches. No model load, no
service impact.

Global scales are pinned to 1.0 so alphas are 1.0 and the only variable under
test is the row ordering.
"""

import sys

import torch

import flashinfer
from flashinfer.fused_moe import cutlass_fused_moe
from vllm.model_executor.layers.quantization.utils.nvfp4_utils import swizzle_blockscale

E, H, I, T, TOPK = 8, 512, 512, 16, 2
DEV = "cuda"

# e2m1: 4-bit float, codes 0-7 positive, 8-15 the negatives.
E2M1 = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
                     -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0])


def quantize(x, swizzled=False):
    """-> (packed uint8, e4m3 block scales). Global scale pinned to 1.

    fp4_quantize hands the scale factors back as raw uint8 holding e4m3 BITS, so
    every consumer has to re-view them; reading them as integers silently
    produces garbage many orders of magnitude too large.
    """
    gs = torch.ones(1, device=x.device, dtype=torch.float32)
    q, sf = flashinfer.fp4_quantize(x, gs, sf_vec_size=16,
                                    is_sf_swizzled_layout=swizzled)
    return q, sf.view(torch.float8_e4m3fn)


def dequantize(q, sf, shape):
    """Exact inverse of the above, in fp32 -- this defines 'correct'."""
    lut = E2M1.to(q.device)
    b = q.reshape(-1).view(torch.uint8)
    lo, hi = b & 0xF, (b >> 4) & 0xF
    vals = torch.stack([lut[lo.long()], lut[hi.long()]], dim=-1).reshape(*shape[:-1], shape[-1])
    scales = sf.view(torch.float8_e4m3fn).reshape(*shape[:-1], shape[-1] // 16).float()
    return (vals.reshape(*shape[:-1], shape[-1] // 16, 16) * scales.unsqueeze(-1)).reshape(shape)


def reference_moe(x, w1, w3, w2, ids, weights):
    """SwiGLU MoE in fp32 over dequantized weights: silu(x@w1^T) * (x@w3^T) @ w2^T."""
    out = torch.zeros(x.shape[0], w2.shape[1], device=x.device, dtype=torch.float32)
    for t in range(x.shape[0]):
        for k in range(ids.shape[1]):
            e = ids[t, k].item()
            g = torch.nn.functional.silu(x[t].float() @ w1[e].float().T)
            u = x[t].float() @ w3[e].float().T
            out[t] += weights[t, k].float() * ((g * u) @ w2[e].float().T)
    return out


def run_kernel(w13_q, w13_sf, w2_q, w2_sf, x, ids, weights):
    ones = torch.ones(1, device=DEV, dtype=torch.float32)
    xq, xsf = flashinfer.fp4_quantize(x, ones, sf_vec_size=16, is_sf_swizzled_layout=True)
    # act-global and alpha entries must be scalar or per-expert; all 1.0 here so
    # the row ordering is the only variable under test.
    ones_e = torch.ones(w13_q.shape[0], device=DEV, dtype=torch.float32)
    quant_scales = [
        ones_e,                                      # a1_gscale
        swizzle_blockscale(w13_sf).view(torch.int32),
        ones_e,                                      # g1_alphas
        ones_e,                                      # a2_gscale
        swizzle_blockscale(w2_sf).view(torch.int32),
        ones_e,                                      # g2_alphas
    ]
    return cutlass_fused_moe(
        input=xq,
        token_selected_experts=ids.to(torch.int),
        token_final_scales=weights,
        fc1_expert_weights=w13_q.view(torch.long),
        fc2_expert_weights=w2_q.view(torch.long),
        output_dtype=torch.bfloat16,
        quant_scales=quant_scales,
        input_sf=xsf,
    )


def main():
    torch.manual_seed(0)
    p = torch.cuda.get_device_properties(0)
    print(f"device={p.name} cc={p.major}.{p.minor}", flush=True)

    w1 = torch.randn(E, I, H, device=DEV).bfloat16() * 0.5   # gate
    w3 = torch.randn(E, I, H, device=DEV).bfloat16() * 0.5   # up
    w2 = torch.randn(E, H, I, device=DEV).bfloat16() * 0.5   # down
    x = torch.randn(T, H, device=DEV).bfloat16()

    # quantize each piece, then dequantize to define the exact reference
    q1, s1 = quantize(w1.reshape(E * I, H)); q3, s3 = quantize(w3.reshape(E * I, H))
    q2, s2 = quantize(w2.reshape(E * H, I))
    d1 = dequantize(q1, s1, (E * I, H)).reshape(E, I, H)
    d3 = dequantize(q3, s3, (E * I, H)).reshape(E, I, H)
    d2 = dequantize(q2, s2, (E * H, I)).reshape(E, H, I)

    ones = torch.ones(1, device=DEV, dtype=torch.float32)
    xq, xsf = flashinfer.fp4_quantize(x, ones, sf_vec_size=16, is_sf_swizzled_layout=False)
    xd = dequantize(xq, xsf.view(torch.float8_e4m3fn), (T, H))

    # Sanity: the dequantizer must invert the quantizer, or "reference" is
    # meaningless and both variants would look equally wrong.
    rt = (xd - x.float()).abs().max().item() / x.float().abs().max().item()
    print(f"quant/dequant round-trip rel error: {rt:.4f}", flush=True)
    if rt > 0.2:
        print("ROUND-TRIP BROKEN -- reference is not trustworthy, aborting.")
        return 1

    ids = torch.stack([torch.randperm(E, device=DEV)[:TOPK] for _ in range(T)]).int()
    weights = torch.rand(T, TOPK, device=DEV, dtype=torch.float32)
    weights = weights / weights.sum(-1, keepdim=True)

    ref = reference_moe(xd, d1, d3, d2, ids, weights)
    print(f"reference computed, |ref|max={ref.abs().max():.4f}", flush=True)

    q1v, q3v = q1.view(E, I, -1), q3.view(E, I, -1)
    s1v, s3v = s1.view(E, I, -1), s3.view(E, I, -1)
    variants = {
        "[gate, up]  (SGLang order)": (torch.cat([q1v, q3v], 1), torch.cat([s1v, s3v], 1)),
        "[up, gate]  (vLLM reorder)": (torch.cat([q3v, q1v], 1), torch.cat([s3v, s1v], 1)),
    }
    print()
    print("EXPERIMENT 1 -- which w13 row order does the kernel want? "
          "(all global scales pinned to 1.0)", flush=True)
    for name, (w13q, w13sf) in variants.items():
        try:
            out = run_kernel(w13q.reshape(E, 2 * I, -1).contiguous(),
                             w13sf.reshape(E, 2 * I, H // 16).contiguous(),
                             q2.view(E, H, -1).contiguous(),
                             s2.reshape(E, H, I // 16).contiguous(),
                             x, ids, weights)
            out = out[0] if isinstance(out, (tuple, list)) else out
            o = out.float()
            err = (o - ref).abs().max().item()
            rel = err / max(ref.abs().max().item(), 1e-6)
            cos = torch.nn.functional.cosine_similarity(
                o.reshape(1, -1), ref.reshape(1, -1)).item()
            print(f"  {name}: max_abs={err:.4e} rel={rel:.4e} cos={cos:.6f}", flush=True)
        except Exception as e:
            print(f"  {name}: FAILED {type(e).__name__}: {str(e)[:220]}", flush=True)
    return 0


# (entry point at the bottom runs both experiments)


# ---------------------------------------------------------------------------
# Experiment 2: non-unit global scales.
#
# Experiment 1 pinned every global scale to 1.0 to isolate the row ordering.
# That deliberately sidesteps the alpha/global-scale plumbing -- which is now
# the prime suspect, since the ordering is exonerated and the kernel itself
# computes correctly on GB10. A real modelopt checkpoint carries per-tensor
# weight_scale_2 and input_scale values, and vLLM folds them into
# g1_alphas / a1_gscale. If that convention is inverted, shapes stay valid and
# the output is silently wrong -- exactly the observed failure.


def experiment_scales():
    torch.manual_seed(0)
    print()
    print("EXPERIMENT 2 -- non-unit global scales (the alpha plumbing)", flush=True)

    # Deliberately awkward magnitudes so a wrong alpha convention cannot cancel.
    wmag, xmag = 0.05, 8.0
    w1 = torch.randn(E, I, H, device=DEV).bfloat16() * wmag
    w3 = torch.randn(E, I, H, device=DEV).bfloat16() * wmag
    w2 = torch.randn(E, H, I, device=DEV).bfloat16() * wmag
    x = torch.randn(T, H, device=DEV).bfloat16() * xmag

    def q_with_gs(t, gs_val):
        gs = torch.full((1,), gs_val, device=DEV, dtype=torch.float32)
        q, sf = flashinfer.fp4_quantize(t, gs, sf_vec_size=16,
                                        is_sf_swizzled_layout=False)
        return q, sf.view(torch.float8_e4m3fn)

    def deq_with_gs(q, sf, shape, gs_val):
        return dequantize(q, sf, shape) / gs_val

    gs_w = 6.0 / (w1.float().abs().max().item())      # map amax -> fp4 max
    gs_a = 6.0 / (x.float().abs().max().item())

    q1, s1 = q_with_gs(w1.reshape(E * I, H), gs_w)
    q3, s3 = q_with_gs(w3.reshape(E * I, H), gs_w)
    q2, s2 = q_with_gs(w2.reshape(E * H, I), gs_w)
    d1 = deq_with_gs(q1, s1, (E * I, H), gs_w).reshape(E, I, H)
    d3 = deq_with_gs(q3, s3, (E * I, H), gs_w).reshape(E, I, H)
    d2 = deq_with_gs(q2, s2, (E * H, I), gs_w).reshape(E, H, I)

    xq_lin, xsf_lin = q_with_gs(x, gs_a)
    xd = deq_with_gs(xq_lin, xsf_lin, (T, H), gs_a)
    rt = (xd - x.float()).abs().max().item() / x.float().abs().max().item()
    print(f"  round-trip rel error with gs_a={gs_a:.4f}: {rt:.4f}", flush=True)

    ids = torch.stack([torch.randperm(E, device=DEV)[:TOPK] for _ in range(T)]).int()
    weights = torch.rand(T, TOPK, device=DEV, dtype=torch.float32)
    weights = weights / weights.sum(-1, keepdim=True)
    ref = reference_moe(xd, d1, d3, d2, ids, weights)

    # winning order from experiment 1: [up, gate]
    q1v, q3v = q1.view(E, I, -1), q3.view(E, I, -1)
    s1v, s3v = s1.view(E, I, -1), s3.view(E, I, -1)
    w13q = torch.cat([q3v, q1v], 1).reshape(E, 2 * I, -1).contiguous()
    w13sf = torch.cat([s3v, s1v], 1).reshape(E, 2 * I, H // 16).contiguous()

    gs = torch.full((1,), gs_a, device=DEV, dtype=torch.float32)
    xq, xsf = flashinfer.fp4_quantize(x, gs, sf_vec_size=16, is_sf_swizzled_layout=True)
    e_ones = torch.ones(E, device=DEV, dtype=torch.float32)

    # alpha = 1/(gs_w * gs_a): undoes both global scales in the kernel epilogue.
    candidates = {
        "alpha=1/(gs_w*gs_a), a_gscale=gs_a": (e_ones / (gs_w * gs_a), e_ones * gs_a),
        "alpha=1/(gs_w*gs_a), a_gscale=1/gs_a": (e_ones / (gs_w * gs_a), e_ones / gs_a),
        "alpha=gs_w*gs_a,     a_gscale=gs_a": (e_ones * (gs_w * gs_a), e_ones * gs_a),
    }
    for name, (alpha, a_gs) in candidates.items():
        try:
            out = cutlass_fused_moe(
                input=xq, token_selected_experts=ids.to(torch.int),
                token_final_scales=weights,
                fc1_expert_weights=w13q.view(torch.long),
                fc2_expert_weights=q2.view(E, H, -1).contiguous().view(torch.long),
                output_dtype=torch.bfloat16,
                quant_scales=[a_gs, swizzle_blockscale(w13sf).view(torch.int32), alpha,
                              a_gs, swizzle_blockscale(
                                  s2.reshape(E, H, I // 16).contiguous()).view(torch.int32),
                              alpha],
                input_sf=xsf)
            out = out[0] if isinstance(out, (tuple, list)) else out
            o = out.float()
            rel = (o - ref).abs().max().item() / max(ref.abs().max().item(), 1e-6)
            cos = torch.nn.functional.cosine_similarity(
                o.reshape(1, -1), ref.reshape(1, -1)).item()
            print(f"  {name}: rel={rel:.4e} cos={cos:.6f}", flush=True)
        except Exception as e:
            print(f"  {name}: FAILED {type(e).__name__}: {str(e)[:180]}", flush=True)


if __name__ == "__main__":
    main()
    experiment_scales()
