"""Lesson A2: roofline math. Pure Python, no torch, so it can be tested anywhere."""
import statistics

# T4 datasheet peaks (NVIDIA): 320 GB/s GDDR6, 65 TFLOPS FP16 tensor cores, 8.1 TFLOPS FP32.
# Real T4s reach less (power/clock limits), which is why we also MEASURE both numbers.
T4_DATASHEET = {"bandwidth_bytes_per_s": 320e9, "fp16_flops": 65e12, "fp32_flops": 8.1e12}


def matmul_flops(m, n, k):
    """(m,k) @ (k,n): one multiply and one add per output element per k."""
    return 2 * m * n * k


def matmul_bytes(m, n, k, bytes_per_element):
    """Minimum traffic: read A and B once, write C once."""
    return (m * k + k * n + m * n) * bytes_per_element


def matmul_intensity(m, n, k, bytes_per_element):
    """Arithmetic intensity in FLOPs per byte moved."""
    return matmul_flops(m, n, k) / matmul_bytes(m, n, k, bytes_per_element)


def ridge_point(peak_flops, peak_bandwidth):
    """Intensity (FLOPs/byte) where the memory roof meets the compute roof."""
    return peak_flops / peak_bandwidth


def attainable_flops(intensity, peak_flops, peak_bandwidth):
    """Roofline: you cannot beat the compute roof or intensity x bandwidth."""
    return min(peak_flops, intensity * peak_bandwidth)


def is_memory_bound(intensity, peak_flops, peak_bandwidth):
    return intensity < ridge_point(peak_flops, peak_bandwidth)


def decode_tokens_per_s_ceiling(weight_bytes, bandwidth, batch=1):
    """Upper bound while memory-bound: each step reads all weights once and yields `batch` tokens.
    Ignores KV-cache reads, activations and compute, so real numbers are lower."""
    return batch * bandwidth / weight_bytes


def model_bandwidth_utilization(steps_per_s, weight_bytes, peak_bandwidth):
    """MBU: fraction of memory bandwidth used just to stream the weights once per decode step."""
    return steps_per_s * weight_bytes / peak_bandwidth


def summarize(values):
    """Report ranges, not single numbers."""
    return {"min": min(values), "median": statistics.median(values), "max": max(values), "n": len(values)}
