# SPDX-License-Identifier: Apache-2.0
"""Supply the activation scale vLLM's NVFP4 MoE reads but never loads.

`LibertAIDAI/GLM-5.3-Flash-NVFP4` is **weight-only** NVFP4: 37,152
`weight_scale` and `weight_scale_2` tensors, `"input_activations": null`, and
**zero `input_scale` tensors**.

vLLM has a weight-only path for linear layers (`ModelOptNvFp4W4A16LinearMethod`),
and both NVFP4 linear methods refuse such a checkpoint loudly:

    raise ValueError("NVFP4 quantization was selected, dynamic quantization is
                      not supported.")

`ModelOptNvFp4FusedMoE` has no weight-only variant and no such guard. It
registers

    w13_input_scale = PerTensorScaleParameter(data=torch.empty(...))

which the checkpoint never fills, and then folds that **uninitialised memory**
into the dequantisation alphas (`g1_alphas = weight_scale_2 * input_scale`) and
the activation global scale (`a1_gscale = 1.0 / input_scale`). The MoE output is
silently wrong; the model degenerates to `"locklocklock…"`.

`marlin` avoids this by dequantising the weights and running a plain bf16 GEMM,
never needing an activation scale — which is why it is reported as "critical for
GB10 correctness". That framing is misleading: the issue is weight-only NVFP4,
not the architecture.

## Why a constant is a legitimate fix, not a fudge

`input_scale` cancels analytically: activations are quantised with
`gscale = 1/input_scale` and dequantised with `alpha ∝ input_scale`. It affects
only how well activations fit fp4's range through the e4m3 block scales, so any
value in a wide safe band is *correct*, not approximate:

  * no clipping requires  `input_scale >= amax_act / 2688`
  * no block-scale underflow requires roughly `input_scale <= amax_block / 0.1`

1.0 sits inside that band for post-norm transformer activations, and it is the
configuration the standalone MoE harness validated on GB10 against an fp32
reference (`cos = 0.9969`, i.e. fp4 noise).

Set `VLLM_GLM53_MOE_INPUT_SCALE` to override the value; unset, this module does
nothing at all.
"""

import os

import torch

from vllm.logger import init_logger

logger = init_logger(__name__)

ENV_VAR = "VLLM_GLM53_MOE_INPUT_SCALE"


def _fill(layer, value: float) -> list[str]:
    filled = []
    for name in ("w13_input_scale", "w2_input_scale"):
        param = getattr(layer, name, None)
        if param is None:
            continue
        with torch.no_grad():
            param.data.fill_(value)
        filled.append(name)
    return filled


def register() -> bool:
    raw = os.environ.get(ENV_VAR, "").strip()
    if not raw:
        return False
    try:
        value = float(raw)
    except ValueError:
        logger.warning("%s=%r is not a float; MoE input-scale fix not applied",
                       ENV_VAR, raw)
        return False
    if not (value > 0.0):
        logger.warning("%s must be > 0, got %s; not applied", ENV_VAR, value)
        return False

    from vllm.model_executor.layers.quantization.modelopt import ModelOptNvFp4FusedMoE

    if getattr(ModelOptNvFp4FusedMoE, "_glm53_input_scale_patched", False):
        return True

    original = ModelOptNvFp4FusedMoE.process_weights_after_loading

    def patched(self, layer):
        # Must run BEFORE the original: it folds these scales into the alphas
        # and then replaces the parameters, so afterwards is too late.
        filled = _fill(layer, value)
        if filled:
            logger.info_once(
                "glm53 MoE fix: set %s to %.6g for weight-only NVFP4 "
                "(checkpoint ships no input_scale; vLLM would otherwise fold "
                "uninitialised memory into the dequantisation alphas)",
                ", ".join(filled), value,
            )
        return original(self, layer)

    ModelOptNvFp4FusedMoE.process_weights_after_loading = patched
    ModelOptNvFp4FusedMoE._glm53_input_scale_patched = True
    logger.info("glm53 MoE fix armed: %s=%s", ENV_VAR, value)
    return True
