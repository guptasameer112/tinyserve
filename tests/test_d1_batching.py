"""D1: no-padding layout, output routing, attention isolation and full-model parity."""
import pytest

torch = pytest.importorskip("torch")

from tinyserve.batching import pack_prompts, split_packed  # noqa: E402
from tinyserve.models.attention import AttentionMetadata, reference_attention  # noqa: E402


@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
def test_hand_layout_and_round_trip_without_mutation(dtype):
    prompts = [torch.tensor([1, 7, 9], dtype=dtype), torch.tensor([1, 12], dtype=dtype),
               torch.tensor([3], dtype=dtype)]
    originals = [prompt.clone() for prompt in prompts]
    flat, metadata, lengths = pack_prompts(prompts)
    assert flat.tolist() == [1, 7, 9, 1, 12, 3] and lengths == (3, 2, 1)
    assert metadata.positions.tolist() == [0, 1, 2, 0, 1, 0]
    assert metadata.sequence_ids.tolist() == [0, 0, 0, 1, 1, 2]
    assert flat.dtype == metadata.positions.dtype == metadata.sequence_ids.dtype == dtype
    assert metadata.cache is None
    assert flat.device == metadata.positions.device == metadata.sequence_ids.device == prompts[0].device
    for original, prompt, restored in zip(originals, prompts, split_packed(flat, lengths)):
        assert torch.equal(original, prompt) and torch.equal(original, restored)
    flat[0] = 5
    assert all(torch.equal(original, prompt) for original, prompt in zip(originals, prompts))


def test_single_request_is_ordinary_causal_layout():
    flat, metadata, lengths = pack_prompts((torch.tensor([1, 7, 9]),))
    assert lengths == (3,) and metadata.positions.tolist() == [0, 1, 2]
    assert torch.equal(metadata.allowed_mask(), AttentionMetadata(torch.arange(3)).allowed_mask())
    assert flat.shape == (3,)


def test_hand_block_diagonal_mask():
    _, metadata, _ = pack_prompts([torch.tensor([1, 2, 3]), torch.tensor([4, 5])])
    assert metadata.allowed_mask().tolist() == [
        [True, False, False, False, False],
        [True, True, False, False, False],
        [True, True, True, False, False],
        [False, False, False, True, False],
        [False, False, False, True, True],
    ]


def test_position_resets_alone_do_not_isolate_requests():
    _, metadata, _ = pack_prompts([torch.tensor([1, 2]), torch.tensor([3])])
    query = torch.ones(3, 1, 1)
    key = torch.zeros_like(query)
    value = torch.tensor([[[1.0]], [[1.0]], [[10.0]]])
    output, weights = reference_attention(query, key, value, metadata.allowed_mask())
    torch.testing.assert_close(output[:, 0, 0], torch.tensor([1.0, 1.0, 10.0]), rtol=0, atol=0)
    mask = metadata.allowed_mask()
    assert torch.count_nonzero(weights[:, ~mask]).item() == 0
    wrong, _ = reference_attention(query, key, value, AttentionMetadata(metadata.positions).allowed_mask())
    assert wrong[0, 0, 0].item() == pytest.approx(5.5)
    assert wrong[2, 0, 0].item() == pytest.approx(5.5)


def test_split_logits_routes_each_last_row_and_returns_views():
    logits = torch.arange(30).reshape(5, 6)
    chunks = split_packed(logits, (3, 2))
    assert [tuple(chunk.shape) for chunk in chunks] == [(3, 6), (2, 6)]
    assert torch.equal(chunks[0][-1], logits[2])
    assert torch.equal(chunks[1][-1], logits[4])
    chunks[1][0, 0] = -1
    assert logits[3, 0].item() == -1


@pytest.mark.parametrize("prompts", [
    [], (), torch.tensor([1]), [None], [torch.tensor(1)], [torch.tensor([[1]])],
    [torch.empty(0, dtype=torch.int64)], [torch.tensor([1.0])], [torch.tensor([True])],
    [torch.tensor([-1])], [torch.tensor([1]), torch.tensor([2], dtype=torch.int32)],
])
def test_invalid_prompts_raise(prompts):
    with pytest.raises(ValueError):
        pack_prompts(prompts)


@pytest.mark.parametrize("lengths", [[], (), [0, 2], [-1, 3], [True, 1], [1.5, 0.5], [1], [3], torch.tensor([2])])
def test_invalid_split_lengths_raise(lengths):
    with pytest.raises(ValueError):
        split_packed(torch.zeros(2, 4), lengths)


@pytest.mark.parametrize("tensor", [None, torch.tensor(1)])
def test_split_requires_token_major_tensor(tensor):
    with pytest.raises(ValueError, match="token dimension"):
        split_packed(tensor, [1])


def pair(kv_heads=2, device="cpu", dtype=torch.float32):
    transformers = pytest.importorskip("transformers")
    from tinyserve.models.llama import LlamaForCausalLM

    torch.manual_seed(11)
    config = transformers.LlamaConfig(
        hidden_size=64, intermediate_size=128, num_hidden_layers=3,
        num_attention_heads=4, num_key_value_heads=kv_heads, vocab_size=64,
        max_position_embeddings=8, eos_token_id=None, pad_token_id=0,
    )
    config._attn_implementation = "eager"
    hf = transformers.LlamaForCausalLM(config).to(device=device, dtype=dtype).eval()
    return LlamaForCausalLM.from_huggingface(hf), hf


@torch.no_grad()
def packed_logits(model, prompts):
    flat, metadata, lengths = pack_prompts(prompts)
    return split_packed(model(flat, metadata.positions, metadata), lengths)


@pytest.mark.parametrize("kv_heads", [4, 2, 1])
@pytest.mark.parametrize("lengths", [(1,), (3, 2, 1), (5, 5)])
def test_full_model_and_hf_parity_for_variable_lengths(kv_heads, lengths):
    from tinyserve.probe import compare_packed_prompts

    ours, hf = pair(kv_heads)
    prompts = [torch.arange(1, length + 1) for length in lengths]
    result = compare_packed_prompts(ours, hf, prompts)
    assert result["request_lengths"] == list(lengths)
    assert result["total_tokens"] == sum(lengths)
    assert result["padded_token_slots"] == len(lengths) * max(lengths)
    assert all(row["all_row_argmax_equal"] for row in result["requests"])
    assert [row["next_token_id"] for row in result["requests"]] == [
        chunk[-1].argmax().item() for chunk in packed_logits(ours, prompts)
    ]


def test_changing_one_request_cannot_change_any_other_request():
    ours, _ = pair()
    prompts = [torch.tensor([1, 7, 9]), torch.tensor([1, 12]), torch.tensor([1, 20])]
    baseline = packed_logits(ours, prompts)
    altered = packed_logits(ours, [prompts[0], torch.tensor([30, 31]), prompts[2]])
    assert torch.equal(baseline[0], altered[0]) and torch.equal(baseline[2], altered[2])
    assert not torch.equal(baseline[1], altered[1])


def test_future_change_stays_causal_inside_its_request():
    ours, _ = pair()
    prompts = [torch.tensor([1, 7, 9]), torch.tensor([1, 12])]
    original = packed_logits(ours, prompts)
    changed = packed_logits(ours, [torch.tensor([1, 7, 20]), prompts[1]])
    assert torch.equal(original[0][:2], changed[0][:2]) and torch.equal(original[1], changed[1])


def test_request_reordering_preserves_outputs_after_routing():
    ours, _ = pair()
    prompts = [torch.tensor([1, 7, 9]), torch.tensor([1]), torch.tensor([1, 12])]
    baseline = packed_logits(ours, prompts)
    order = [2, 0, 1]
    reordered = packed_logits(ours, [prompts[index] for index in order])
    for chunk, index in zip(reordered, order):
        torch.testing.assert_close(chunk, baseline[index], rtol=1e-5, atol=1e-6)


def test_contiguous_cache_still_rejects_packed_requests():
    ours, _ = pair()
    flat, metadata, _ = pack_prompts([torch.tensor([1, 7]), torch.tensor([1])])
    metadata.cache = ours.new_kv_cache(8)
    with torch.no_grad(), pytest.raises(ValueError, match="one request"):
        ours(flat, metadata.positions, metadata)


def test_probe_rejects_oversized_individual_request_not_total_packed_tokens():
    from tinyserve.probe import compare_packed_prompts

    ours, hf = pair()
    with pytest.raises(ValueError, match="each prompt"):
        compare_packed_prompts(ours, hf, [torch.arange(1, 10)])
    result = compare_packed_prompts(ours, hf, [torch.arange(1, 7), torch.arange(1, 7)])
    assert result["total_tokens"] == 12 > ours.config.max_position_embeddings


@pytest.mark.parametrize("invalid_mode", ["training", "float16", "hf_attention"])
def test_probe_rejects_nonreference_modes(invalid_mode):
    from tinyserve.probe import compare_packed_prompts

    ours, hf = pair()
    if invalid_mode == "training":
        ours.train()
    elif invalid_mode == "float16":
        ours.half()
    else:
        hf.config._attn_implementation = "sdpa"
    with pytest.raises(ValueError):
        compare_packed_prompts(ours, hf, [torch.tensor([1, 7])])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_cuda_layout_and_mixed_devices_rejection():
    prompts = [torch.tensor([1, 7], device="cuda", dtype=torch.int32),
               torch.tensor([1], device="cuda", dtype=torch.int32)]
    flat, metadata, _ = pack_prompts(prompts)
    assert flat.device == metadata.positions.device == metadata.sequence_ids.device == prompts[0].device
    with pytest.raises(ValueError, match="same device"):
        pack_prompts([prompts[0], prompts[1].cpu()])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_fp16_packed_logits_against_same_rounded_weights_in_fp32():
    from tinyserve.models.llama import LlamaForCausalLM

    low, hf = pair(device="cuda", dtype=torch.float16)
    high = LlamaForCausalLM(hf.config, device="cuda", dtype=torch.float32).eval()
    high.load_state_dict({name: weight.float() for name, weight in low.state_dict().items()})
    prompts = [torch.tensor([1, 7, 9], device="cuda"), torch.tensor([1, 12], device="cuda")]
    for actual, reference in zip(packed_logits(low, prompts), packed_logits(high, prompts)):
        torch.testing.assert_close(actual, reference, rtol=1e-2, atol=1e-2)
