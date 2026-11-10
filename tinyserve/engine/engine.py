"""D5: single-threaded continuous generation over the D4 scheduler."""
from dataclasses import dataclass
from typing import Optional, Tuple

import torch

from tinyserve.engine.block_manager import _positive_integer
from tinyserve.engine.memory import plan_kv_memory
from tinyserve.engine.scheduler import Scheduler, _nonnegative_integer
from tinyserve.generation import _validate_generation_request
from tinyserve.models.attention import AttentionMetadata
from tinyserve.sampling import sample_token


@dataclass(frozen=True)
class RequestOutput:
    request_id: int
    token_ids: Tuple[int, ...]
    generated_tokens: Tuple[int, ...]
    finish_reason: Optional[str]

    @property
    def finished(self):
        return self.finish_reason is not None


@dataclass
class _Request:
    ids: torch.Tensor
    prompt_tokens: int
    max_new_tokens: int
    temperature: float
    top_k: int
    top_p: float
    generator: Optional[torch.Generator]
    eos_ids: Tuple[int, ...]
    finish_reason: Optional[str] = None


class InferenceEngine:
    """One model, one fixed pool and one pending forward; no server or threading.

    add_request/cancel are allowed between successful steps. step returns
    cumulative snapshots only for requests that sampled a token in that step.
    Forward failures before cache commit retain the D4 plan for retry.
    """
    def __init__(self, model, kv_budget_bytes, block_size=16, max_batch_tokens=128):
        if model.training:
            raise ValueError("engine requires a model in eval mode")
        max_batch_tokens = _positive_integer("max_batch_tokens", max_batch_tokens)
        self.model = model
        self.memory_budget = plan_kv_memory(model, kv_budget_bytes, block_size)
        self.cache = model.new_managed_paged_cache(self.memory_budget.num_blocks, block_size)
        self.scheduler = Scheduler(self.cache, max_batch_tokens)
        self._requests = {}
        self._pending_logits = None
        if self.cache.allocated_bytes != self.memory_budget.allocated_bytes:
            raise RuntimeError("allocated KV bytes differ from the planned budget")

    @classmethod
    def from_available_memory(cls, model, reserve_bytes, block_size=16, max_batch_tokens=128):
        """CUDA only: sample driver free bytes AFTER loading weights, minus explicit headroom.

        Headroom covers activations, logits, gather temporaries and other users.
        This snapshot is not a guarantee: allocations and workloads can change.
        """
        reserve_bytes = _positive_integer("reserve_bytes", reserve_bytes)
        device = model.get_input_embeddings().weight.device
        if device.type != "cuda":
            raise ValueError("available-memory sizing requires a CUDA model")
        free_bytes, _ = torch.cuda.mem_get_info(device)
        if reserve_bytes >= free_bytes:
            raise ValueError("workspace reserve leaves no free memory for KV blocks")
        return cls(model, free_bytes - reserve_bytes, block_size, max_batch_tokens)

    def _check_model(self):
        weight = self.model.get_input_embeddings().weight
        if self.model.training or weight.device != self.cache.key.device or weight.dtype != self.cache.key.dtype:
            raise ValueError("engine model must remain in eval mode on its original device/dtype")

    def _require_idle(self):
        self._check_model()
        self.scheduler._require_idle()

    @property
    def has_pending_requests(self):
        return any(request.finish_reason is None for request in self._requests.values())

    def add_request(self, request_id, input_ids, max_new_tokens=20, temperature=0.0,
                    top_k=0, top_p=1.0, generator=None):
        """Clone a prompt; preserve C2 EOS, sampling and strict context-limit behavior."""
        self._require_idle()
        request_id = _nonnegative_integer("request_id", request_id)
        if request_id > torch.iinfo(torch.int64).max:
            raise ValueError("request ID must fit an int64 sequence ID")
        if request_id in self._requests:
            raise ValueError("request ID has already been used")
        eos_ids = _validate_generation_request(
            self.model, input_ids, max_new_tokens, temperature, top_k, top_p, generator,
        )
        ids = input_ids.clone()
        if max_new_tokens:
            self.scheduler.add_request(request_id, ids.numel(), max_new_tokens - 1)
        self._requests[request_id] = _Request(
            ids, ids.numel(), int(max_new_tokens), temperature, top_k, top_p, generator, eos_ids,
            None if max_new_tokens else "length",
        )
        return self.get_output(request_id)

    def get_output(self, request_id):
        request_id = _nonnegative_integer("request_id", request_id)
        if request_id not in self._requests:
            raise ValueError("unknown request ID")
        request = self._requests[request_id]
        ids = tuple(request.ids.tolist())
        return RequestOutput(request_id, ids, ids[request.prompt_tokens:], request.finish_reason)

    def cancel(self, request_id):
        self._require_idle()
        output = self.get_output(request_id)
        if output.finished:
            raise ValueError("request is already finished or cancelled")
        self.scheduler.cancel(request_id)
        self._requests[request_id].finish_reason = "cancelled"
        return self.get_output(request_id)

    def _rng_snapshots(self, owners):
        generators = {}
        for owner in owners:
            request = self._requests[owner]
            if request.temperature:
                generators[id(request.generator)] = request.generator
        snapshots = []
        for generator in generators.values():
            if generator is None:
                device = self.cache.key.device
                state = (torch.cuda.get_rng_state(device) if device.type == "cuda"
                         else torch.random.get_rng_state())
            else:
                state = generator.get_state()
            snapshots.append((generator, state))
        return snapshots

    @torch.no_grad()
    def step(self) -> Tuple[RequestOutput, ...]:
        """Schedule -> packed forward -> sample -> acknowledge -> release EOS requests.

        Replays and partial prompts do not sample. Saved logits allow sampling
        failures to propagate without rerunning an already committed forward.
        Sampling errors restore RNG state and leave token histories unchanged.
        Invalid saved logits require discarding the engine, not blind forward retries.
        """
        self._check_model()
        plan = self.scheduler.schedule()
        if plan is None:
            return ()
        if self._pending_logits is None:
            ids = torch.cat([self._requests[item.request_id].ids[item.start:item.end]
                             for item in plan.work])
            positions = torch.cat([torch.arange(item.start, item.end, device=ids.device, dtype=torch.int64)
                                   for item in plan.work])
            owners = torch.cat([positions.new_full((item.end - item.start,), item.request_id)
                                for item in plan.work])
            self._pending_logits = self.model(
                ids, positions, AttentionMetadata(positions, owners, paged_cache=self.cache),
            )
        logits = self._pending_logits
        vocab = self.model.get_input_embeddings().num_embeddings
        if (not isinstance(logits, torch.Tensor) or logits.shape != (plan.num_tokens, vocab)
                or not logits.is_floating_point() or logits.device != self.cache.key.device):
            raise ValueError("model must return floating-point logits (tokens, vocab) on its device")
        candidates, offset = {}, 0
        for item in plan.work:
            offset += item.end - item.start
            request = self._requests[item.request_id]
            if item.phase != "recompute" and item.end >= request.prompt_tokens:
                candidates[item.request_id] = logits[offset - 1]
        snapshots = self._rng_snapshots(candidates)
        updates = {}
        try:
            for owner, scores in candidates.items():
                request = self._requests[owner]
                token = sample_token(scores, request.temperature, request.top_k,
                                     request.top_p, request.generator)
                new_ids = torch.cat((request.ids, request.ids.new_tensor([token])))
                reason = ("eos" if token in request.eos_ids else
                          "length" if new_ids.numel() - request.prompt_tokens == request.max_new_tokens
                          else None)
                updates[owner] = (new_ids, reason)
        except (ValueError, RuntimeError):
            for generator, state in snapshots:
                if generator is not None:
                    generator.set_state(state)
                elif self.cache.key.device.type == "cuda":
                    torch.cuda.set_rng_state(state, self.cache.key.device)
                else:
                    torch.random.set_rng_state(state)
            raise
        self.scheduler.complete_step()
        self._pending_logits = None
        for owner, (ids, reason) in updates.items():
            request = self._requests[owner]
            request.ids, request.finish_reason = ids, reason
            if reason == "eos" and self.scheduler.requests[owner].status != "finished":
                self.scheduler.cancel(owner)
        return tuple(self.get_output(owner) for owner in updates)

    def run(self) -> Tuple[RequestOutput, ...]:
        """Drain finite queued work; use step() instead to admit requests between forwards."""
        while self.has_pending_requests:
            self.step()
        return tuple(self.get_output(owner) for owner in self._requests)
