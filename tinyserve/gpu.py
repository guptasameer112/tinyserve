"""Lesson A2: GPU measurements (CUDA only). Every tensor takes its device/dtype from the arguments."""
import torch

from tinyserve.roofline import matmul_bytes, matmul_flops, matmul_intensity, summarize


def time_cuda_ms(fn, warmup=5, iters=20):
    """Average milliseconds per call of fn, timed with CUDA events over `iters` calls."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) / iters


def repeat_time_cuda_ms(fn, trials=7, warmup=5, iters=20):
    """`trials` independent timings, so we can report a range (the T4 varies run to run)."""
    return [time_cuda_ms(fn, warmup=warmup, iters=iters) for _ in range(trials)]


def measure_bandwidth_gbs(num_bytes, device, trials=7):
    """Device-to-device copy: reads num_bytes and writes num_bytes. Returns GB/s per trial."""
    src = torch.empty(num_bytes, dtype=torch.uint8, device=device)
    dst = torch.empty_like(src)
    times_ms = repeat_time_cuda_ms(lambda: dst.copy_(src), trials=trials)
    return [2 * num_bytes / (t * 1e-3) / 1e9 for t in times_ms]


def measure_bandwidth_curve(sizes_mib, device, trials=7):
    rows = []
    for size in sizes_mib:
        stats = summarize(measure_bandwidth_gbs(size * 2**20, device, trials))
        rows.append({"size_mib": size, "gb_s": stats})
    return rows


def measure_matmul_tflops(m, n, k, device, dtype, trials=7):
    """Square-ish matmul throughput. Returns TFLOPS per trial."""
    a = torch.randn(m, k, device=device, dtype=dtype)
    b = torch.randn(k, n, device=device, dtype=dtype)
    times_ms = repeat_time_cuda_ms(lambda: a @ b, trials=trials)
    return [matmul_flops(m, n, k) / (t * 1e-3) / 1e12 for t in times_ms]


def measure_matmul_curve(sizes, device, dtype, trials=7):
    bytes_per_element = torch.empty(0, dtype=dtype).element_size()
    rows = []
    for size in sizes:
        stats = summarize(measure_matmul_tflops(size, size, size, device, dtype, trials))
        rows.append({
            "size": size,
            "tflops": stats,
            "intensity": matmul_intensity(size, size, size, bytes_per_element),
        })
    return rows


def measure_decode_curve(batches, k, n, device, dtype, trials=7):
    """x (B, k) @ W (k, n): what one linear layer does in a decode step with B sequences.
    Returns per-batch median time, achieved GB/s (bytes moved / time) and TFLOPS."""
    weight = torch.randn(k, n, device=device, dtype=dtype)
    bytes_per_element = weight.element_size()
    rows = []
    for batch in batches:
        x = torch.randn(batch, k, device=device, dtype=dtype)
        times_ms = repeat_time_cuda_ms(lambda: x @ weight, trials=trials)
        median_ms = summarize(times_ms)["median"]
        seconds = median_ms * 1e-3
        rows.append({
            "batch": batch,
            "median_ms": median_ms,
            "intensity": matmul_intensity(batch, n, k, bytes_per_element),
            "gb_s": matmul_bytes(batch, n, k, bytes_per_element) / seconds / 1e9,
            "tflops": matmul_flops(batch, n, k) / seconds / 1e12,
            "rows_per_s": batch / seconds,
        })
    return rows
