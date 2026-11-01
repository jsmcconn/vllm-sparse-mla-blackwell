# vllm-sparse-mla-blackwell

**A LibertAI Labs artefact.**

A vLLM plugin that makes **NoPE sparse-MLA models run on Blackwell consumer and
workstation GPUs** (sm_120 and sm_121), and fixes a silent correctness fault in
vLLM's NVFP4 mixture-of-experts path for weight-only checkpoints.

Built to serve [`LibertAIDAI/GLM-5.3-Flash-NVFP4`](https://huggingface.co/LibertAIDAI/GLM-5.3-Flash-NVFP4),
which vLLM could previously not run on these GPUs at all. Nothing in it is
specific to that checkpoint.

## What it fixes

**1. No MLA path for NoPE geometry on sm_120/121.** A model with
`qk_rope_head_dim = 0` is locked out of vLLM on these GPUs: every MLA *prefill*
backend rejects its head dimensions, and the only sparse *decode* backend requires
the packed `fp8_ds_mla` layout, whose cache kernel asserts `pe_dim == 64`. This
package supplies a hand-written sparse-MLA CUDA kernel plus a vLLM attention
backend that takes `head_size = kv_lora_rank` natively, so no fabricated rope
block is needed.

**2. Uninitialised activation scale in the NVFP4 MoE.**
`ModelOptNvFp4FusedMoE` registers `w13_input_scale` as
`PerTensorScaleParameter(data=torch.empty(...))` and expects the checkpoint to fill
it. A weight-only NVFP4 checkpoint (`"input_activations": null`) never does, so the
value stays at its uninitialised contents, observed as **0.0**. The dequantisation
alpha becomes `weight_scale_2 * 0` and **every expert output is multiplied by
zero**, with no error and no warning. Reported upstream as
[vllm-project/vllm#54189](https://github.com/vllm-project/vllm/issues/54189).

Either fault alone makes a model emit one token repeatedly, which is why they were
hard to separate: changing attention backends never changes the output while the
experts are zeroed.

## Verified hardware

Every combination below passes a full case matrix against an fp32 reference, run as
the installed artefact rather than a dev build.

| GPU | arch | verified |
| --- | --- | --- |
| NVIDIA GB10 (DGX Spark class) | sm_121 | yes, and serving production |
| NVIDIA RTX PRO 6000 Blackwell Server | sm_120 | yes |
| NVIDIA GeForce RTX 5090 | sm_120 | yes |

| axis | supported |
| --- | --- |
| heads per rank | 8, 16, 32, so TP=8, TP=4 and TP=2 on a 64-head model |
| KV dtype | bfloat16 and fp8_e4m3 |
| prefill and decode | one code path, mixed batches in a single launch |
| CUDA graphs | `AttentionCGSupport.UNIFORM_BATCH`, so spec-decode shapes capture |

Worst relative error across all of it is 4.1e-3 with per-row cosine at or above
0.999998, which is bf16 noise. For scale, the TileLang kernel this replaces
measures up to 3.8e-3 against the same reference.

## Install

```bash
GLM53_ARCHS=121a pip install --no-build-isolation --target /opt/ext .   # GB10
GLM53_ARCHS=120a pip install --no-build-isolation --target /opt/ext .   # RTX Blackwell
```

Then point vLLM at it and enable the two hooks. Both are `vllm.general_plugins`
entry points, so **no vLLM file is patched**, and both are inert unless their
environment variable is set:

```sh
PYTHONPATH=/opt/ext
VLLM_GLM53_CUDA_SPARSE_MLA=1     # the sparse-MLA backend
VLLM_GLM53_MOE_INPUT_SCALE=1.0   # the missing activation scale
```

Serve with `--kv-cache-dtype bfloat16` or `fp8`.

## Design notes worth knowing

- **Causality belongs to the indexer, not the kernel.** The only mask is an
  `index >= 0` sentinel. That makes the index axis layout-agnostic, which is what
  lets vLLM's paged physical slot ids be passed straight through, and it is why
  prefill and decode share one launch.
- **Sized for the real shared-memory budget.** Datacenter Blackwell gives about
  227 KB of opt-in shared memory per block; sm_120/121 gives 101,376 bytes. The
  tiles are chosen against the smaller number rather than inherited from a
  datacenter kernel.
- **No tcgen05, no TMA, no wgmma.** Those are datacenter Blackwell features. This
  uses the warp-level `mma.sync.m16n8k16` path with `ldmatrix` for every operand.
- **Two bugs fixed relative to the reference kernel it replaces**: an
  out-of-bounds read on the mask sentinel, and an all-NaN result for a token whose
  entire top-k selection is masked.

## Performance, including what did not work

On 2x GB10 at concurrency 1, medians of five runs, serving GLM-5.3-Flash:

| configuration | tok/s | KV tokens |
| --- | --- | --- |
| no speculative decoding | 14.46 | 127,291 at 8K context |
| MTP with 3 draft tokens | 24.04 | |
| MTP plus CUDA graphs, vLLM's default breakable graphs | 23.69 | 27,852 |
| MTP plus CUDA graphs, `VLLM_USE_BREAKABLE_CUDAGRAPH=0` | 24.38 | 37,273 |
| MTP, eager, fp8 KV | 24.21 | 88,790 at 64K context |

Without speculation the lane sits on the memory-bandwidth wall for an 18B-active
model, so the attention kernel is not the limiter at concurrency 1. CUDA graphs as
vLLM configures them by default are a net loss, giving no speed and cutting the KV
cache by a factor of 4.6.

fp8 KV is **slower than bf16**, 0.88x to 0.96x, because the per-element dequantise
costs more than the halved gather traffic saves. It closes to 0.96x at larger KV
pools, which is the memory-bound regime that matters in production, so the trade
is roughly 4 percent of throughput for double the cache.

## Limits

The backend passes the indexer's full selection to attention, including the
incomplete K-pool tail and masked padding. After installation, run
`PYTHONPATH=/opt/ext python tests/run_backend.py` to check this backend path
against the mean of all selected rows and verify nested text configurations.

- 64 heads on a single rank is not built. It would need 114,944 bytes of shared
  memory against the 101,376 byte ceiling, so TP must be at least 2 on a 64-head
  model.
- At 8 heads per rank the MMA pads to its 16-row minimum, so half the tensor core
  is wasted. That is the price of TP=8.
- The kernel is bf16 or fp8_e4m3 only. The packed `fp8_ds_mla` record is not
  supported, and is not needed.

## License

Apache-2.0.
