"""D3: a CPU-side free list and reference counts for physical KV blocks."""
from numbers import Integral


def _positive_integer(name, value):
    if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
        raise ValueError("%s must be a positive integer" % name)
    return int(value)


class BlockPoolExhausted(RuntimeError):
    """The pool cannot satisfy a reservation without reclaiming live blocks."""


class BlockAllocator:
    """Single-threaded block leases. Each allocation starts with one reference."""
    def __init__(self, num_blocks):
        self.num_blocks = _positive_integer("num_blocks", num_blocks)
        self._free = list(reversed(range(self.num_blocks)))
        self._ref_counts = [0] * self.num_blocks

    @property
    def free_blocks(self):
        """Snapshot in next-allocation order."""
        return tuple(reversed(self._free))

    @property
    def ref_counts(self):
        return tuple(self._ref_counts)

    @property
    def num_free_blocks(self):
        return len(self._free)

    def allocate(self, count):
        """Reserve all requested blocks or raise without changing any state."""
        count = _positive_integer("count", count)
        if count > self.num_free_blocks:
            raise BlockPoolExhausted(
                "need %d blocks but only %d are free" % (count, self.num_free_blocks)
            )
        blocks = tuple(self._free.pop() for _ in range(count))
        for block in blocks:
            self._ref_counts[block] = 1
        return blocks

    def _live_block(self, block):
        if (isinstance(block, bool) or not isinstance(block, Integral)
                or not 0 <= block < self.num_blocks):
            raise ValueError("physical block ID is out of range")
        if self._ref_counts[block] == 0:
            raise ValueError("block has no live reference")
        return int(block)

    def retain(self, block):
        block = self._live_block(block)
        self._ref_counts[block] += 1

    def release(self, block):
        block = self._live_block(block)
        self._ref_counts[block] -= 1
        if self._ref_counts[block] == 0:
            self._free.append(block)
