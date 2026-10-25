"""Lesson A2: roofline diagram as code. Needs matplotlib."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def plot_roofline(peak_flops, peak_bandwidth, points, path, title="Roofline", burst_flops=None):
    """points: list of (label, intensity FLOPs/byte, achieved FLOP/s). Saves a PNG to `path`.
    burst_flops (optional) draws a dashed line for the short-burst compute peak."""
    ridge = peak_flops / peak_bandwidth
    x_values = [10 ** (e / 10) for e in range(-10, 41)]
    roof = [min(peak_flops, x * peak_bandwidth) / 1e12 for x in x_values]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.loglog(x_values, roof, color="black", label="roofline (peaks)")
    ax.axvline(ridge, color="gray", linestyle=":", label="ridge %.0f FLOPs/byte" % ridge)
    if burst_flops:
        ax.axhline(burst_flops / 1e12, color="tab:red", linestyle="--", linewidth=1, label="burst peak (before throttling)")
    for label, intensity, flops in points:
        ax.scatter([intensity], [flops / 1e12], zorder=3)
        ax.annotate(label, (intensity, flops / 1e12), textcoords="offset points", xytext=(4, 4), fontsize=8)
    ax.set_xlabel("arithmetic intensity (FLOPs / byte)")
    ax.set_ylabel("achieved TFLOPS")
    ax.set_title(title)
    ax.legend(loc="lower right")
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path
