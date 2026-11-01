"""Check the installed vLLM backend, including padded indexer selections."""

from types import SimpleNamespace

import torch
import glm53_sparse_mla
from glm53_sparse_mla.backend import (
    Glm53SparseMLAImpl,
    Glm53SparseMLAMetadata,
    Glm53SparseMLAMetadataBuilder,
)
from vllm.v1.attention.backend import AttentionType


def check_indexer_tail():
    q = torch.zeros((1, 32, 512), dtype=torch.bfloat16, device="cuda")
    cache = torch.zeros((1, 256, 512), dtype=q.dtype, device=q.device)
    indices = torch.full((1, 2176), -1, dtype=torch.int32, device=q.device)
    indices[0, 0] = 0
    indices[0, 2048:2051] = torch.arange(1, 4, device=q.device)
    metadata = Glm53SparseMLAMetadata(
        num_reqs=1, max_query_len=1, max_seq_len=256,
        num_actual_tokens=1,
        query_start_loc=torch.tensor([0, 1], dtype=torch.int32, device=q.device),
        slot_mapping=torch.tensor([0], dtype=torch.int64, device=q.device),
        block_table=torch.tensor([[0]], dtype=torch.int32, device=q.device),
        req_id_per_token=torch.tensor([0], dtype=torch.int32, device=q.device),
        block_size=256, num_decodes=1, num_decode_tokens=1,
    )
    impl = Glm53SparseMLAImpl(
        num_heads=32, head_size=512, scale=256 ** -0.5, num_kv_heads=1,
        alibi_slopes=None, sliding_window=None, kv_cache_dtype="bfloat16",
        logits_soft_cap=None, attn_type=AttentionType.DECODER,
        kv_sharing_target_layer_name=None, topk_indices_buffer=indices,
        kv_lora_rank=512,
    )
    values = torch.arange(4, device=q.device, dtype=q.dtype)[:, None].expand(4, 512).contiguous()
    impl.do_kv_cache_update(
        values, torch.empty((4, 1, 0), dtype=q.dtype, device=q.device), cache,
        torch.arange(4, dtype=torch.int64, device=q.device), "bfloat16",
        torch.ones((), device=q.device),
    )
    torch.testing.assert_close(cache[0, :4], values, rtol=0, atol=0)
    impl.record_logical_topk_ready()
    out, _ = impl.forward_mqa(q, cache, metadata, SimpleNamespace())
    # Zero queries give uniform attention over the four selected rows, including the tail.
    expected = cache[0, :4].float().mean(0).expand_as(out)
    torch.testing.assert_close(out.float(), expected, rtol=0, atol=0)
    print("padded indexer tail: PASS")


def check_metadata_builder():
    text_config = SimpleNamespace(index_topk=2048, index_kpool=4)
    config = SimpleNamespace(
        model_config=SimpleNamespace(
            hf_config=SimpleNamespace(text_config=text_config),
            hf_text_config=text_config,
        ),
        scheduler_config=SimpleNamespace(max_num_batched_tokens=1),
    )
    Glm53SparseMLAMetadataBuilder(
        SimpleNamespace(block_size=256), [], config, torch.device("cuda")
    )
    print("nested text config: PASS")


if __name__ == "__main__":
    print("installed package:", glm53_sparse_mla.__file__)
    check_indexer_tail()
    check_metadata_builder()
