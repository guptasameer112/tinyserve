"""C2: single-request generation with our flattened model; no KV cache yet."""
from numbers import Integral

import torch

from tinyserve.models.attention import AttentionMetadata
from tinyserve.sampling import sample_token, validate_sampling_options


@torch.no_grad()
def generate(model, input_ids, max_new_tokens=20, temperature=0.0, top_k=0, top_p=1.0, generator=None):
    """Return prompt + new token IDs (including generated EOS), without mutating input.

    Default is greedy. EOS comes from model.config (integer, list/tuple or None).
    A prompt EOS does not stop generation. Prompt + requested new tokens must
    fit max_position_embeddings; oversized requests raise, never truncate.
    """
    validate_sampling_options(temperature, top_k, top_p)
    if isinstance(max_new_tokens, bool) or not isinstance(max_new_tokens, Integral) or max_new_tokens < 0:
        raise ValueError("max_new_tokens must be a nonnegative integer")
    if model.training:
        raise ValueError("generation requires a model in eval mode")
    if input_ids.ndim != 1 or input_ids.numel() == 0 or input_ids.dtype not in (torch.int32, torch.int64):
        raise ValueError("input_ids must be a nonempty integer (N,) tensor")
    embedding = model.get_input_embeddings()
    if input_ids.device != embedding.weight.device:
        raise ValueError("input_ids must be on the model device")
    vocab_size = embedding.num_embeddings
    if bool((input_ids < 0).any()) or bool((input_ids >= vocab_size).any()):
        raise ValueError("input_ids must be within the model vocabulary")
    if input_ids.numel() + max_new_tokens > model.config.max_position_embeddings:
        raise ValueError("prompt plus max_new_tokens exceeds the model context limit")
    if generator is not None:
        if not isinstance(generator, torch.Generator) or generator.device != input_ids.device:
            raise ValueError("generator must be a torch.Generator on the input_ids device")
    eos = model.config.eos_token_id
    if eos is None:
        eos_ids = ()
    elif isinstance(eos, Integral) and not isinstance(eos, bool):
        eos_ids = (int(eos),)
    elif isinstance(eos, (list, tuple)):
        eos_ids = tuple(eos)
    else:
        raise ValueError("eos_token_id must be an integer, list/tuple of integers, or None")
    if any(isinstance(token, bool) or not isinstance(token, Integral) or not 0 <= token < vocab_size
           for token in eos_ids):
        raise ValueError("EOS token IDs must be integers within the model vocabulary")

    ids = input_ids.clone()
    for _ in range(max_new_tokens):
        positions = torch.arange(ids.numel(), device=ids.device)
        logits = model(ids, positions, AttentionMetadata(positions))
        if logits.ndim != 2 or logits.shape != (ids.numel(), vocab_size):
            raise ValueError("model must return logits with shape (tokens, vocab)")
        token = sample_token(logits[-1], temperature, top_k, top_p, generator)
        ids = torch.cat((ids, ids.new_tensor([token])))
        if token in eos_ids:
            break
    return ids
