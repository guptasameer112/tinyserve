"""D4: bounded FCFS scheduling with release-and-recompute preemption."""
from dataclasses import dataclass, replace
from numbers import Integral
from types import MappingProxyType
from typing import Tuple

from tinyserve.engine.block_manager import _positive_integer


def _nonnegative_integer(name, value):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
        raise ValueError("%s must be a nonnegative integer" % name)
    return int(value)


@dataclass(frozen=True)
class RequestState:
    request_id: int
    prompt_tokens: int
    decode_steps: int
    processed_tokens: int = 0
    status: str = "waiting"
    preemptions: int = 0

    @property
    def total_tokens(self):
        return self.prompt_tokens + self.decode_steps


@dataclass(frozen=True)
class ScheduledWork:
    request_id: int
    start: int
    end: int
    phase: str


@dataclass(frozen=True)
class BatchPlan:
    work: Tuple[ScheduledWork, ...]
    preempted: Tuple[int, ...]

    @property
    def num_tokens(self):
        return sum(item.end - item.start for item in self.work)


class Scheduler:
    """Own an initially empty managed cache; callers execute and complete plans.

    Arrival order defines priority. Prefill/recomputation use the remaining token
    budget; decode processes one token per request per step. Only newer resident
    requests can be preempted. Token IDs stay with the caller for replay.
    """
    def __init__(self, cache, max_batch_tokens):
        self.max_batch_tokens = _positive_integer("max_batch_tokens", max_batch_tokens)
        if cache.block_tables or cache.lengths:
            raise ValueError("scheduler requires an empty managed cache")
        self.cache = cache
        self._requests = {}
        self.requests = MappingProxyType(self._requests)
        self._pending = None
        self._expected_lengths = None

    def _check_cache(self):
        resident = {owner for owner, state in self._requests.items() if state.status == "running"}
        if set(self.cache.block_tables) != resident or set(self.cache.lengths) != resident:
            raise RuntimeError("cache ownership changed outside the scheduler")
        for owner in resident:
            if not 0 <= self.cache.lengths[owner] <= self._requests[owner].processed_tokens:
                raise RuntimeError("cache length changed outside a scheduled forward")

    def _require_idle(self):
        if self._pending is not None:
            raise RuntimeError("complete the pending step before changing requests")
        self._check_cache()

    def add_request(self, request_id, prompt_tokens, decode_steps=0):
        """Queue finite work. decode_steps counts processed appends, not sampled outputs."""
        self._require_idle()
        request_id = _nonnegative_integer("request_id", request_id)
        prompt_tokens = _positive_integer("prompt_tokens", prompt_tokens)
        decode_steps = _nonnegative_integer("decode_steps", decode_steps)
        if request_id in self._requests:
            raise ValueError("request ID has already been used")
        total = prompt_tokens + decode_steps
        if total > self.cache.model.config.max_position_embeddings:
            raise ValueError("request exceeds the model context limit")
        if total > self.cache.num_blocks * self.cache.block_size:
            raise ValueError("request cannot fit in the entire block pool")
        self._requests[request_id] = RequestState(request_id, prompt_tokens, decode_steps)

    def cancel(self, request_id):
        """Cancel waiting, running or preempted work and release any owned blocks."""
        self._require_idle()
        request_id = _nonnegative_integer("request_id", request_id)
        if request_id not in self._requests:
            raise ValueError("unknown request ID")
        state = self._requests[request_id]
        if state.status in ("finished", "cancelled"):
            raise ValueError("request is already finished or cancelled")
        if request_id in self.cache.block_tables:
            self.cache.release_request(request_id)
        self._requests[request_id] = replace(state, status="cancelled")

    def schedule(self):
        """Reserve a batch, or return the existing pending plan unchanged for retry."""
        if self._pending is not None:
            return self._pending
        self._check_cache()
        active = [owner for owner, state in self._requests.items()
                  if state.status not in ("finished", "cancelled")]
        if not active:
            return None
        remaining = self.max_batch_tokens
        work, preempted = [], []
        for index, owner in enumerate(active):
            if remaining == 0:
                break
            state = self._requests[owner]
            start = self.cache.lengths.get(owner, 0)
            if start < state.processed_tokens:
                phase, stop = "recompute", state.processed_tokens
            elif start < state.prompt_tokens:
                phase, stop = "prefill", state.prompt_tokens
            else:
                phase, stop = "decode", start + 1
            end = min(stop, start + remaining)
            required = (end + self.cache.block_size - 1) // self.cache.block_size
            extra = required - len(self.cache.block_tables.get(owner, ()))
            available = self.cache.num_free_blocks
            victims = []
            for candidate in reversed(active[index + 1:]):
                if available >= extra:
                    break
                if candidate in self.cache.block_tables:
                    victims.append(candidate)
                    available += len(self.cache.block_tables[candidate])
            if available < extra:
                # Earlier requests keep priority; do not evict anyone for an unfit plan.
                break
            for victim in victims:
                self.cache.release_request(victim)
                previous = self._requests[victim]
                self._requests[victim] = replace(
                    previous, status="preempted", preemptions=previous.preemptions + 1,
                )
                preempted.append(victim)
            self.cache.reserve(owner, end)
            self._requests[owner] = replace(state, status="running")
            work.append(ScheduledWork(owner, start, end, phase))
            remaining -= end - start
        if not work:
            raise RuntimeError("no request can make progress despite single-request admission checks")
        self._pending = BatchPlan(tuple(work), tuple(preempted))
        self._expected_lengths = dict(self.cache.lengths)
        for item in work:
            self._expected_lengths[item.request_id] = item.end
        return self._pending

    def complete_step(self):
        """Acknowledge one successful whole-batch forward, then release finished work."""
        if self._pending is None:
            raise RuntimeError("there is no pending step")
        if (dict(self.cache.lengths) != self._expected_lengths
                or set(self.cache.block_tables) != set(self._expected_lengths)):
            raise RuntimeError("the pending forward has not committed the expected cache lengths")
        for item in self._pending.work:
            state = self._requests[item.request_id]
            processed = max(state.processed_tokens, item.end)
            status = "finished" if processed == state.total_tokens else "running"
            self._requests[item.request_id] = replace(state, processed_tokens=processed, status=status)
            if status == "finished":
                self.cache.release_request(item.request_id)
        self._pending = None
        self._expected_lengths = None
