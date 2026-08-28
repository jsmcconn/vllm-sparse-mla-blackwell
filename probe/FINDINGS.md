# Why vLLM produces `locklocklock…` for GLM-5.3-Flash-NVFP4

Established with `moe_reorder_probe.py` on a live GB10 — **no model load, no
service impact**. The harness builds a tiny NVFP4 MoE, dequantizes it to get an
exact fp32 reference, and drives `flashinfer.fused_moe.cutlass_fused_moe`
directly.

## What the harness eliminated

**1. The kernel is fine on sm_121.** Driven correctly it matches the reference to
`rel = 7.3e-02, cos = 0.9969` — which is fp4 noise (the quantize/dequantize
round-trip alone is 14.9%). So "FLASHINFER_CUTLASS is broken on GB10" is not
literally true; it is broken *as vLLM drives it*.

**2. vLLM's `reorder_w1w3_to_w3w1` is CORRECT** — this was my hypothesis and the
harness killed it:

```
[gate, up]  (SGLang order): rel=8.38e-01  cos=0.5237   <- wrong
[up, gate]  (vLLM reorder): rel=7.85e-02  cos=0.9969   <- correct
```

**3. The alpha convention vLLM uses is CORRECT:**

```
alpha=1/(gs_w*gs_a), a_gscale=gs_a  : rel=7.35e-02 cos=0.9969   <- correct
alpha=1/(gs_w*gs_a), a_gscale=1/gs_a: rel=2.46e+01 cos=0.9963   <- right shape, wrong magnitude
alpha=gs_w*gs_a,     a_gscale=gs_a  : rel=2.91e+03 cos=0.7513   <- garbage
```

Note the middle row: an inverted activation scale keeps cosine at 0.996 while the
magnitude is off by ~25×. Mis-scaled MoE output looks *structurally* right and
still destroys the model — which is what a silent-garbage bug looks like.

## The actual cause: the checkpoint is weight-only, and vLLM's NVFP4 MoE is not

`LibertAIDAI/GLM-5.3-Flash-NVFP4` is **weight-only**:

```
*.weight_scale    : 37152 tensors
*.weight_scale_2  : 37152 tensors
*.input_scale     : 0 tensors
quantization_config: {"input_activations": null, "quant_algo": "NVFP4", ...}
```

`weight_scale_2 ≈ 3.8e-5` confirms the inverse convention vLLM expects, and block
scales topping out at exactly 448 (e4m3 max) is by design. **The checkpoint is
correct.**

vLLM's `modelopt.py` has a weight-only path for *linear* layers —
`ModelOptNvFp4W4A16LinearMethod` (:1227) — and both NVFP4 linear methods guard
the case explicitly:

```python
raise ValueError("NVFP4 quantization was selected, dynamic quantization is not supported.")
```

**There is no W4A16 equivalent for MoE.** `ModelOptNvFp4FusedMoE` (:1372) has no
such guard, registers `w13_input_scale` / `w2_input_scale` as
`PerTensorScaleParameter`s the checkpoint never fills, and then derives the
global scale from them anyway (`:1199 input_global_scale = layer.input_scale.max()`).
The alphas handed to the kernel are therefore computed from an unpopulated
parameter, and the MoE output is silently wrong.

## Why `marlin` is the documented fix — and why that framing misleads

HF discussion #5 records `marlin` as *"critical for GB10 correctness"*. Marlin
dequantizes the NVFP4 weights and runs an ordinary bf16 GEMM, so it never needs
an activation scale and never touches this code path.

**So the fix is not GB10-specific — it is weight-only-NVFP4-specific.** Any
weight-only modelopt NVFP4 checkpoint should hit this on any arch, B200 included.
That is a testable prediction, and it reframes the bug from "sm_121 kernel gap" to
"vLLM cannot serve weight-only NVFP4 MoE".

## Upstream

The defect is that the linear path refuses this checkpoint loudly while the MoE
path accepts it and produces wrong numbers. Either behaviour would be fine; the
mismatch is the bug. The fix is a W4A16 MoE path (dynamic per-tensor activation
scale, which is what SGLang does) or, failing that, the same guard the linear
methods already have.
