"""Lesson A2 tests. Roofline math is exact. GPU measurements are sanity checks only (they skip without CUDA)."""
import pytest

from tinyserve.roofline import (
    T4_DATASHEET, attainable_flops, decode_tokens_per_s_ceiling, is_memory_bound,
    matmul_bytes, matmul_flops, matmul_intensity, ridge_point, summarize,
)


def test_matmul_flops_and_bytes():
    assert matmul_flops(2, 3, 4) == 2 * 2 * 3 * 4
    assert matmul_bytes(2, 3, 4, 2) == (2 * 4 + 4 * 3 + 2 * 3) * 2


def test_decode_matmul_intensity_is_about_batch():
    # (B, K) @ (K, N) with K=N=4096 in FP16: bytes are dominated by the weight, so intensity ~ B.
    # Activations add B*(K+N) elements to the weight's K*N, so intensity is slightly below B:
    # B / (1 + B*(K+N)/(K*N)) = 62.1 at B=64, which is within 5%.
    for batch in (1, 8, 64):
        assert matmul_intensity(batch, 4096, 4096, 2) == pytest.approx(batch, rel=0.05)


def test_ridge_and_attainable():
    peak_flops, peak_bw = 65e12, 320e9
    assert ridge_point(peak_flops, peak_bw) == pytest.approx(203.125)
    assert attainable_flops(10, peak_flops, peak_bw) == pytest.approx(10 * 320e9)
    assert attainable_flops(1000, peak_flops, peak_bw) == peak_flops
    assert is_memory_bound(1, peak_flops, peak_bw)
    assert not is_memory_bound(1000, peak_flops, peak_bw)


def test_decode_ceiling_scales_with_batch():
    one = decode_tokens_per_s_ceiling(2.2e9, 240e9, batch=1)
    assert one == pytest.approx(240e9 / 2.2e9)
    assert decode_tokens_per_s_ceiling(2.2e9, 240e9, batch=8) == pytest.approx(8 * one)


def test_summarize_reports_a_range():
    stats = summarize([3.0, 1.0, 2.0])
    assert (stats["min"], stats["median"], stats["max"], stats["n"]) == (1.0, 2.0, 3.0, 3)


def test_t4_datasheet_values():
    assert T4_DATASHEET["bandwidth_bytes_per_s"] == 320e9


try:
    import torch
except ImportError:  # pure-math tests above still run without torch
    torch = None
needs_cuda = pytest.mark.skipif(
    torch is None or not torch.cuda.is_available(), reason="needs torch and a CUDA GPU"
)


@needs_cuda
def test_bandwidth_is_positive_and_below_datasheet():
    from tinyserve.gpu import measure_bandwidth_gbs

    values = measure_bandwidth_gbs(256 * 2**20, "cuda", trials=3)
    assert len(values) == 3
    assert all(0 < v < 1.05 * T4_DATASHEET["bandwidth_bytes_per_s"] / 1e9 for v in values)


@needs_cuda
def test_fp16_matmul_beats_fp32_matmul():
    from tinyserve.gpu import measure_matmul_tflops

    fp16 = summarize(measure_matmul_tflops(2048, 2048, 2048, "cuda", torch.float16, trials=3))["median"]
    fp32 = summarize(measure_matmul_tflops(2048, 2048, 2048, "cuda", torch.float32, trials=3))["median"]
    assert fp16 > fp32


@needs_cuda
def test_decode_curve_fields_and_batching_helps():
    from tinyserve.gpu import measure_decode_curve

    rows = measure_decode_curve([1, 16], 2048, 5632, "cuda", torch.float16, trials=3)
    assert {"batch", "median_ms", "intensity", "gb_s", "tflops", "rows_per_s"} <= set(rows[0])
    # 16x the rows should cost far less than 16x the time while memory-bound.
    assert rows[1]["median_ms"] < 4 * rows[0]["median_ms"]
