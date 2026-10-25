"""Lesson A1: tokens -> tensors -> logits.

Flattened-token convention (used by the whole project): a batch of text is one 1-D tensor of
token ids with shape (N,). Here N is simply the length of one prompt.
"""
import torch

MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"


def load_model(device="cuda", dtype=torch.float16, model_id=MODEL_ID, quiet=True):
    """Load tokenizer and Hugging Face model in eval mode."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if quiet:
        from tinyserve.quiet import quiet_logs

        quiet_logs()
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype).to(device).eval()
    return tokenizer, model


def encode(tokenizer, text, device):
    """Text -> 1-D LongTensor (N,). The tokenizer prepends BOS for Llama models."""
    ids = tokenizer(text, return_tensors="pt").input_ids[0]
    return ids.to(device)


@torch.no_grad()
def forward_logits(model, input_ids):
    """input_ids (N,) -> logits (N, vocab). Row i scores the token that comes AFTER position i."""
    return model(input_ids[None, :]).logits[0]


def next_token_greedy(logits):
    """Greedy choice from the last row of logits (N, vocab). Returns a Python int."""
    return int(logits[-1].argmax())


@torch.no_grad()
def generate_greedy_naive(model, input_ids, max_new_tokens):
    """Teaser loop: re-runs the WHOLE sequence every step (no KV cache). Returns ids (N + new,)."""
    ids = input_ids
    for _ in range(max_new_tokens):
        token = next_token_greedy(forward_logits(model, ids))
        ids = torch.cat([ids, torch.tensor([token], device=ids.device, dtype=ids.dtype)])
    return ids


@torch.no_grad()
def prefix_logit_diff(model, input_ids, extra_ids):
    """Max |logits difference| on the shared prefix after appending extra_ids. Causality => ~0."""
    base = forward_logits(model, input_ids)
    longer = forward_logits(model, torch.cat([input_ids, extra_ids.to(input_ids.device)]))
    return (longer[: len(input_ids)].float() - base.float()).abs().max().item()


def shape_report(tokenizer, model, text):
    """Shapes and sizes for the lesson tables. Runs one forward pass."""
    param = next(model.parameters())
    ids = encode(tokenizer, text, param.device)
    logits = forward_logits(model, ids)
    cfg = model.config
    return {
        "text": text,
        "tokens": tokenizer.convert_ids_to_tokens(ids.tolist()),
        "input_ids_shape": tuple(ids.shape),
        "embedding_shape": (len(ids), cfg.hidden_size),
        "logits_shape": tuple(logits.shape),
        "logits_dtype": str(logits.dtype),
        "logits_bytes": logits.numel() * logits.element_size(),
        "num_params": sum(p.numel() for p in model.parameters()),
        "weight_bytes": sum(p.numel() * p.element_size() for p in model.parameters()),
    }
